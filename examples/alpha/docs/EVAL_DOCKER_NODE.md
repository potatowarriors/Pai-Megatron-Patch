# 에이전틱 벤치 실행 노드 — gpu06 DinD 컨테이너 (2026-08-29)

SWE-bench·Terminal-Bench 등 **태스크마다 컨테이너를 빌드·실행하는** 하니스를 돌리기 위한
외부 docker 호스트. Backend.AI 세션(main1/sub1)은 컨테이너 안이라 `cap_sys_admin` 부재 +
seccomp 차단으로 docker 기동이 불가능하다(2026-08-29 실측, [SFT_BENCHMARKS.md](SFT_BENCHMARKS.md) §4).
그래서 docker를 직접 띄울 수 있는 별도 서버 위에 privileged 컨테이너를 만들고,
그 안에서 Docker-in-Docker(DinD)로 벤치 하니스를 실행한다.

**이 문서 하나로** 접속·복구·운영이 되도록 유지한다. 상태는 [STATUS.md](STATUS.md).

## 0. TL;DR

- 서버: `gpu06` (외부 `210.122.96.117:10000` = 앞단 방화벽의 호스트 ssh 포워딩). 계정 `dongholee`.
- 평가 컨테이너: `alpha-eval` (privileged, DinD 동작 확인). 내부망 IP `172.17.0.17`.
- 접속: main1/sub1 → **ProxyJump(gpu06 경유)** → 컨테이너. 전용 키·config는 NFS에 있어 두 노드 공유.
  ```bash
  ssh -F /home/work/vidsearch/.ssh-keys/config alpha-eval
  ```
- **재시작 시 dockerd 수동 기동 필요** (§4).

## 1. 왜 이 구조인가 (설계 근거)

| 문제 | 해법 |
|---|---|
| Backend.AI 세션에서 docker 불가 (capability/seccomp) | docker 되는 외부 서버 gpu06 사용 |
| gpu06 앞단 방화벽이 **선별 포트만** 포워딩 → 컨테이너에 `-p` 열어도 밖에서 안 닿음 (10032는 리스닝 중인데도 외부 거부됨을 실측) | 컨테이너 포트 노출 포기. 이미 열린 **호스트 ssh(10000)를 ProxyJump 발판**으로 삼아 내부망 IP로 직접 진입 |
| 컨테이너 안에서 또 컨테이너(SWE 태스크)를 만들어야 함 | `--privileged` + `-v alpha-eval-docker:/var/lib/docker`(overlayfs 유지, vfs 폴백 방지) |
| main1·sub1이 같은 키로 접속해야 함 (sub1 vLLM→컨테이너 역터널 예정) | 키·config를 NFS `vidsearch/.ssh-keys/`에 배치 |

## 2. 접속 자산 (NFS, 재부팅에도 유지)

```
/home/work/vidsearch/.ssh-keys/
├── id_dockerhost         # 전용 ed25519 개인키 (comment: alpha-eval@backendai-cluster)
├── id_dockerhost.pub
├── config                # gpu06(hop1) + alpha-eval(hop2, ProxyJump) 정의
└── known_hosts
```

- 공개키는 **gpu06 호스트 계정**(`~dongholee/.ssh/authorized_keys`)과 **컨테이너 root**
  양쪽에 등록돼 있다. hop1은 호스트, hop2는 컨테이너.
- 이 키는 이 용도 전용이다. 폐기하려면 양쪽 authorized_keys에서 해당 줄 제거.

## 3. 검증 결과 (2026-08-29)

| 항목 | 결과 |
|---|---|
| main1 → gpu06 → alpha-eval | ✅ 2홉 ProxyJump |
| sub1 → alpha-eval | ✅ (동일 NFS 키/config) |
| 컨테이너 내 DinD | ✅ dockerd 29.1.3, **storage=overlayfs**, cgroup v1 |
| 이미지 pull·실행 | ✅ `hello-world`, `python:3.11-slim`(186MB) |
| 아웃바운드 | ✅ Docker Hub pull OK (DNS 203.248.116.7). ※ 컨테이너에 curl 없음, dockerd 경로는 정상 |

**서버 사양**: CPU 64코어 / RAM 503GB(available 466GB) / `/var/lib/docker` 11TB 중
**1.1TB 여유**(91% 사용 중 — 공용 서버, 아래 주의) / 커널 5.4.0 / Ubuntu 24.04.

## 4. 운영 — 재시작 후 dockerd 복구

컨테이너는 `--restart unless-stopped`라 서버·데몬 재기동 시 자동으로 다시 뜨지만,
**컨테이너 내부 dockerd는 자동 기동되지 않는다** (sshd 자동기동 wrapper는 시스템 바이너리
교체라 안전상 배제). 평가 시작 전 헬스체크로 흡수하는 것이 원칙이고, 수동 복구는:

```bash
ssh -F /home/work/vidsearch/.ssh-keys/config alpha-eval \
  'pgrep -x dockerd >/dev/null || nohup dockerd >/var/log/dockerd.log 2>&1 &'
sleep 8
ssh -F /home/work/vidsearch/.ssh-keys/config alpha-eval 'docker info --format "{{.ServerVersion}} {{.Driver}}"'
```

## 5. 재구축 절차 (컨테이너가 삭제된 경우)

gpu06 호스트에서 (한 줄씩; 긴 값은 변수로 빼 붙여넣기 안전하게):

```bash
K='AAAAC3NzaC1lZDI1NTE5AAAAIPMZakB7lG/jvCn74bjlSPfMomzajNHcIlsPRWR/WQ1e'
PK="ssh-ed25519 $K alpha-eval"
LOOP='while :; do mkdir -p /run/sshd; /usr/sbin/sshd 2>/dev/null; sleep 60; done'
mkdir -p ~/.ssh; chmod 700 ~/.ssh
grep -qF "$K" ~/.ssh/authorized_keys || echo "$PK" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
docker rm -f alpha-eval
docker run -d --name alpha-eval --privileged --restart unless-stopped \
  -v alpha-eval-docker:/var/lib/docker -e PUBKEY="$PK" ubuntu:24.04 bash -c "$LOOP"
docker exec alpha-eval apt-get update -q
docker exec -e DEBIAN_FRONTEND=noninteractive alpha-eval apt-get install -qy openssh-server docker.io procps
docker exec alpha-eval mkdir -p /root/.ssh
docker exec alpha-eval bash -c 'echo "$PUBKEY" > /root/.ssh/authorized_keys'
docker exec alpha-eval chmod 700 /root/.ssh
docker exec alpha-eval chmod 600 /root/.ssh/authorized_keys
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' alpha-eval
```

- 마지막이 출력하는 IP가 이전(172.17.0.17)과 다르면 **`.ssh-keys/config`의 `Host alpha-eval`
  HostName을 갱신**한다.
- 그다음 §4로 dockerd 기동.

## 6. 모델 연결 (SWE/T-Bench 실행 시)

하니스(컨테이너)가 alpha 모델(sub1 vLLM :8000)을 호출해야 한다. 컨테이너→sub1 직접
라우팅이 안 되면 **sub1이 컨테이너로 역터널**을 연다:

```bash
# sub1에서 실행 — 컨테이너의 localhost:8000 이 sub1 vLLM 을 가리키게 됨
ssh -F /home/work/vidsearch/.ssh-keys/config -N -R 8000:localhost:8000 alpha-eval
```

하니스에는 OpenAI-호환 base_url `http://localhost:8000/v1` 로 준다.

## 7. 주의

- **공용 서버다.** gpu06에는 타 사용자 컨테이너 20+개가 상시 가동 중(`docker ps` 확인).
  우리 것은 `alpha-eval` + 이름에 접두 없는 SWE 태스크 컨테이너뿐 — **남의 컨테이너·볼륨·이미지
  절대 건드리지 않는다.** 정리는 `docker ... --filter name=alpha` 또는 SWE 하니스 자체 정리 경로로만.
- `/var/lib/docker` 여유 1.1TB는 공용 소비로 변동한다. SWE-bench Verified 전체 이미지(~100–200GB)
  투입 전 `df -h /var/lib/docker` 재확인. 여유 부족 시 사용자에게 알림(호스트 디스크는 우리가 못 늘림).
- 이 문서의 키는 **공개키**다. 개인키는 NFS의 600 퍼미션 파일에만 있고 문서·git에 넣지 않는다.
