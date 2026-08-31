#!/usr/bin/env python3
"""open-webui 0.11.2 스키마 초기화 — 순환 import 회피.

왜 필요한가 (2026-08-31 실측):
    open_webui/config.py 는 80 행에서 run_migrations() 를 부른다. alembic 이 로드하는
    migrations/env.py 가 models.calendar → utils.automations → events →
    retrieval/web/utils.py 를 타고 되돌아와 config.py **1103 행**의
    ENABLE_LOCAL_WEB_FETCH 를 import 한다 — 아직 정의되지 않았으므로 ImportError.
    run_migrations() 는 이 예외를 log.exception 으로 삼키고 넘어가서 **테이블이 하나도
    생성되지 않은 채** 기동이 계속되고, 나중에 `no such table: config` 로 터진다.

해법:
    ENABLE_DB_MIGRATIONS=false 로 config 로드 중 마이그레이션 재진입을 막고,
    config 를 **끝까지** 로드한 뒤 alembic 을 별도로 돌린다. 되돌아온 import 가
    이미 완성된 sys.modules 항목을 만나므로 순환이 성립하지 않는다.

멱등: `alembic upgrade head` 는 이미 head 면 아무것도 하지 않는다.
"""
import os
import sys

os.environ["ENABLE_DB_MIGRATIONS"] = "false"

import open_webui.config  # noqa: F401,E402  — 여기서 끝까지 로드되는 것이 핵심
from open_webui.env import OPEN_WEBUI_DIR  # noqa: E402

from alembic import command  # noqa: E402
from alembic.config import Config as AlembicConfig  # noqa: E402

cfg = AlembicConfig(OPEN_WEBUI_DIR / "alembic.ini")
cfg.set_main_option("script_location", str(OPEN_WEBUI_DIR / "migrations"))
command.upgrade(cfg, "head")

# 검증 — 침묵한 실패를 여기서 잡는다
from open_webui.internal.db import engine  # noqa: E402
from sqlalchemy import inspect  # noqa: E402

tables = set(inspect(engine).get_table_names())
required = {"config", "auth", "user", "chat"}
missing = required - tables
if missing:
    print(f"❌ 마이그레이션 후에도 테이블 누락: {sorted(missing)}", file=sys.stderr)
    sys.exit(1)
print(f"✅ 스키마 OK — 테이블 {len(tables)}개 (config/auth/user/chat 확인)")
