#!/usr/bin/env python3
"""
Alpha Checkpoint Backup System

Automatically backs up Megatron distributed checkpoints to MinIO object storage.

Usage:
    python checkpoint_backup.py [--config config.yaml] [--dry-run] [--force] [--list]

Environment Variables Required:
    MINIO_ENDPOINT: MinIO server endpoint (e.g., http://minio.example.com:9000)
    MINIO_ACCESS_KEY: MinIO access key
    MINIO_SECRET_KEY: MinIO secret key

Optional:
    MINIO_SECURE: Use HTTPS (default: false)
    MINIO_REGION: Region (default: us-east-1)

Examples:
    # Dry run to see what would be backed up
    python checkpoint_backup.py --dry-run

    # Force re-backup of all checkpoints
    python checkpoint_backup.py --force

    # List current backup state
    python checkpoint_backup.py --list
"""

import os
import sys
import json
import logging
import argparse
import fcntl
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError


@dataclass
class CheckpointInfo:
    """Information about a single checkpoint."""
    experiment_name: str
    iteration: int
    local_path: Path
    total_size: int
    file_count: int
    is_complete: bool
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def size_human(self) -> str:
        """Return human-readable size."""
        size = self.total_size
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"


@dataclass
class BackupState:
    """Persistent state for tracking backups."""
    backed_up_checkpoints: Dict[str, Dict[str, str]] = field(default_factory=dict)
    last_run: Optional[str] = None
    version: str = "1.0"


class CheckpointBackupManager:
    """Manages backup of Megatron distributed checkpoints to MinIO."""

    def __init__(self, config_path: str, dry_run: bool = False, force: bool = False):
        self.script_dir = Path(__file__).parent.resolve()
        self.config = self._load_config(config_path)
        self.dry_run = dry_run
        self.force = force
        self.logger = self._setup_logging()
        self.s3_client = None  # Lazy initialization
        self.state = self._load_state()
        self.lock_fd = None

    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration."""
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = self.script_dir / config_file

        with open(config_file) as f:
            return yaml.safe_load(f)

    def _resolve_path(self, path: str) -> Path:
        """Resolve relative paths based on script directory."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.script_dir / p

    def _setup_logging(self) -> logging.Logger:
        """Configure logging with file and console handlers."""
        logger = logging.getLogger("checkpoint_backup")
        logger.setLevel(self.config["backup"]["logging"]["log_level"])

        # Clear existing handlers
        logger.handlers.clear()

        # File handler with daily rotation
        log_dir = self._resolve_path(self.config["backup"]["logging"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"backup_{datetime.now().strftime('%Y%m%d')}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(console_handler)

        return logger

    def _create_s3_client(self):
        """Create boto3 S3 client configured for MinIO."""
        endpoint = os.environ.get("MINIO_ENDPOINT")
        access_key = os.environ.get("MINIO_ACCESS_KEY")
        secret_key = os.environ.get("MINIO_SECRET_KEY")
        secure = os.environ.get("MINIO_SECURE", "false").lower() == "true"
        region = os.environ.get("MINIO_REGION", "us-east-1")

        if not all([endpoint, access_key, secret_key]):
            raise EnvironmentError(
                "Missing required environment variables: "
                "MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY"
            )

        self.logger.info(f"Connecting to MinIO at {endpoint}")

        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            use_ssl=secure,
            config=BotoConfig(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "adaptive"}
            )
        )

    def _get_s3_client(self):
        """Get or create S3 client (lazy initialization)."""
        if self.s3_client is None:
            self.s3_client = self._create_s3_client()
        return self.s3_client

    def _load_state(self) -> BackupState:
        """Load backup state from persistent storage."""
        state_file = self._resolve_path(self.config["backup"]["state"]["state_file"])
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                    return BackupState(**data)
            except (json.JSONDecodeError, TypeError) as e:
                self.logger.warning(f"Failed to load state file, starting fresh: {e}")
        return BackupState()

    def _save_state(self):
        """Persist backup state."""
        state_file = self._resolve_path(self.config["backup"]["state"]["state_file"])
        state_file.parent.mkdir(parents=True, exist_ok=True)

        with open(state_file, "w") as f:
            json.dump(asdict(self.state), f, indent=2)

    def discover_checkpoints(self) -> List[CheckpointInfo]:
        """Discover all complete checkpoints that need backup."""
        base_path = Path(self.config["backup"]["source"]["base_path"])
        pattern = self.config["backup"]["source"]["experiment_pattern"]
        checkpoints = []

        if not base_path.exists():
            self.logger.warning(f"Base path does not exist: {base_path}")
            return checkpoints

        for exp_dir in sorted(base_path.glob(pattern)):
            if not exp_dir.is_dir():
                continue

            ckpt_dir = exp_dir / self.config["backup"]["source"]["checkpoint_dir"]
            if not ckpt_dir.exists():
                self.logger.debug(f"No checkpoints directory in {exp_dir.name}")
                continue

            # Read latest iteration indicator
            indicator_file = ckpt_dir / self.config["backup"]["source"]["completion_indicator"]
            if not indicator_file.exists():
                self.logger.debug(f"No completion indicator in {exp_dir.name}")
                continue

            try:
                latest_iter = int(indicator_file.read_text().strip())
            except ValueError as e:
                self.logger.warning(f"Invalid iteration indicator in {exp_dir.name}: {e}")
                continue

            # Find all checkpoint directories
            for iter_dir in sorted(ckpt_dir.glob("iter_*")):
                if not iter_dir.is_dir():
                    continue

                try:
                    iteration = int(iter_dir.name.split("_")[1])
                except (IndexError, ValueError):
                    self.logger.warning(f"Invalid checkpoint directory name: {iter_dir.name}")
                    continue

                is_complete = iteration <= latest_iter

                # Calculate total size and file count
                total_size = sum(f.stat().st_size for f in iter_dir.rglob("*") if f.is_file())
                file_count = sum(1 for f in iter_dir.rglob("*") if f.is_file())

                checkpoints.append(CheckpointInfo(
                    experiment_name=exp_dir.name,
                    iteration=iteration,
                    local_path=iter_dir,
                    total_size=total_size,
                    file_count=file_count,
                    is_complete=is_complete
                ))

        return checkpoints

    def filter_pending_backups(self, checkpoints: List[CheckpointInfo]) -> List[CheckpointInfo]:
        """Filter out already backed up checkpoints."""
        if self.force:
            self.logger.info("Force mode: will re-backup all checkpoints")
            return [c for c in checkpoints if c.is_complete]

        pending = []
        for ckpt in checkpoints:
            if not ckpt.is_complete:
                self.logger.debug(f"Skipping incomplete: {ckpt.experiment_name}/{ckpt.local_path.name}")
                continue

            exp_state = self.state.backed_up_checkpoints.get(ckpt.experiment_name, {})
            if ckpt.local_path.name in exp_state:
                self.logger.debug(f"Already backed up: {ckpt.experiment_name}/{ckpt.local_path.name}")
                continue

            pending.append(ckpt)

        return pending

    def upload_file(self, local_path: Path, s3_key: str, bucket: str) -> Tuple[bool, Optional[str]]:
        """Upload a single file to MinIO with multipart support."""
        config = self.config["backup"]["upload"]
        transfer_config = TransferConfig(
            multipart_threshold=config["multipart_threshold"],
            multipart_chunksize=config["multipart_chunksize"],
            max_concurrency=config["max_concurrency"],
            use_threads=True
        )

        for attempt in range(config["retry_attempts"]):
            try:
                if self.dry_run:
                    self.logger.debug(f"[DRY-RUN] Would upload: {local_path.name} -> s3://{bucket}/{s3_key}")
                    return True, None

                # Upload with multipart support
                self._get_s3_client().upload_file(
                    str(local_path),
                    bucket,
                    s3_key,
                    Config=transfer_config
                )

                # Verify upload if enabled
                if self.config["backup"]["verification"]["verify_after_upload"]:
                    response = self._get_s3_client().head_object(Bucket=bucket, Key=s3_key)
                    remote_size = response["ContentLength"]
                    local_size = local_path.stat().st_size

                    if remote_size != local_size:
                        raise ValueError(
                            f"Size mismatch: local={local_size}, remote={remote_size}"
                        )

                return True, None

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                self.logger.warning(
                    f"Upload attempt {attempt + 1}/{config['retry_attempts']} failed "
                    f"({error_code}): {local_path.name}"
                )
                if attempt < config["retry_attempts"] - 1:
                    time.sleep(config["retry_delay"] * (attempt + 1))  # Exponential backoff
                else:
                    return False, str(e)
            except Exception as e:
                self.logger.error(f"Unexpected error uploading {local_path.name}: {e}")
                return False, str(e)

        return False, "Max retries exceeded"

    def backup_checkpoint(self, checkpoint: CheckpointInfo) -> bool:
        """Backup a single checkpoint to MinIO."""
        bucket = self.config["backup"]["destination"]["bucket"]
        prefix = self.config["backup"]["destination"]["prefix"]
        s3_prefix = f"{prefix}/{checkpoint.experiment_name}/{checkpoint.local_path.name}"

        self.logger.info(
            f"Backing up {checkpoint.experiment_name}/{checkpoint.local_path.name} "
            f"({checkpoint.size_human()}, {checkpoint.file_count} files)"
        )

        files_to_upload = [f for f in checkpoint.local_path.rglob("*") if f.is_file()]

        if not files_to_upload:
            self.logger.warning(f"No files found in checkpoint: {checkpoint.local_path}")
            return False

        success_count = 0
        failed_files = []

        # Use thread pool for parallel file uploads
        max_parallel = self.config["backup"]["upload"]["max_parallel_files"]

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {}
            for file_path in files_to_upload:
                relative_path = file_path.relative_to(checkpoint.local_path)
                s3_key = f"{s3_prefix}/{relative_path}"
                futures[executor.submit(self.upload_file, file_path, s3_key, bucket)] = file_path

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    success, error = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_files.append((file_path.name, error))
                        self.logger.error(f"Failed to upload {file_path.name}: {error}")
                except Exception as e:
                    failed_files.append((file_path.name, str(e)))
                    self.logger.error(f"Exception uploading {file_path.name}: {e}")

        if failed_files:
            self.logger.error(
                f"Checkpoint backup incomplete: {len(failed_files)}/{len(files_to_upload)} files failed"
            )
            return False

        self.logger.info(
            f"Successfully backed up {success_count} files for "
            f"{checkpoint.experiment_name}/{checkpoint.local_path.name}"
        )
        return True

    def acquire_lock(self) -> bool:
        """Acquire exclusive lock for backup process."""
        lock_file = self._resolve_path(self.config["backup"]["state"]["lock_file"])
        lock_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.lock_fd = open(lock_file, "w")
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_fd.write(f"{os.getpid()}\n{datetime.now().isoformat()}")
            self.lock_fd.flush()
            return True
        except (IOError, OSError):
            self.logger.warning("Another backup process is running (lock held)")
            if self.lock_fd:
                self.lock_fd.close()
                self.lock_fd = None
            return False

    def release_lock(self):
        """Release exclusive lock."""
        if self.lock_fd:
            try:
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
                self.lock_fd.close()
            except Exception as e:
                self.logger.warning(f"Error releasing lock: {e}")
            finally:
                self.lock_fd = None

    def ensure_bucket_exists(self):
        """Create bucket if it doesn't exist."""
        bucket = self.config["backup"]["destination"]["bucket"]
        try:
            self._get_s3_client().head_bucket(Bucket=bucket)
            self.logger.debug(f"Bucket exists: {bucket}")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404" or error_code == "NoSuchBucket":
                if not self.dry_run:
                    self._get_s3_client().create_bucket(Bucket=bucket)
                    self.logger.info(f"Created bucket: {bucket}")
                else:
                    self.logger.info(f"[DRY-RUN] Would create bucket: {bucket}")
            else:
                raise

    def list_state(self):
        """Display current backup state."""
        print("\n=== Alpha Checkpoint Backup State ===\n")
        print(f"Last run: {self.state.last_run or 'Never'}")
        print(f"State version: {self.state.version}")
        print()

        if not self.state.backed_up_checkpoints:
            print("No checkpoints have been backed up yet.")
        else:
            for exp_name, checkpoints in sorted(self.state.backed_up_checkpoints.items()):
                print(f"Experiment: {exp_name}")
                for ckpt_name, s3_path in sorted(checkpoints.items()):
                    print(f"  - {ckpt_name}: {s3_path}")
                print()

        # Show pending checkpoints
        print("\n=== Pending Checkpoints ===\n")
        all_checkpoints = self.discover_checkpoints()
        pending = self.filter_pending_backups(all_checkpoints)

        if not pending:
            print("No new checkpoints pending backup.")
        else:
            total_size = sum(c.total_size for c in pending)
            print(f"Found {len(pending)} checkpoint(s) pending backup:")
            for ckpt in pending:
                print(f"  - {ckpt.experiment_name}/{ckpt.local_path.name} ({ckpt.size_human()})")
            print(f"\nTotal size to backup: {pending[0].size_human() if len(pending) == 1 else f'{total_size / (1024**3):.2f} GB'}")

    def run(self) -> int:
        """Main backup execution."""
        if not self.acquire_lock():
            return 1

        try:
            # Discover and filter checkpoints
            all_checkpoints = self.discover_checkpoints()
            self.logger.info(f"Discovered {len(all_checkpoints)} total checkpoint(s)")

            pending = self.filter_pending_backups(all_checkpoints)
            self.logger.info(f"Found {len(pending)} checkpoint(s) pending backup")

            if not pending:
                self.logger.info("No new checkpoints to backup")
                return 0

            # Sort by experiment name and iteration
            pending.sort(key=lambda x: (x.experiment_name, x.iteration))

            # Show summary
            total_size = sum(c.total_size for c in pending)
            self.logger.info(f"Total size to backup: {total_size / (1024**3):.2f} GB")

            if self.dry_run:
                self.logger.info("=== DRY RUN MODE - No actual uploads ===")
                for ckpt in pending:
                    self.logger.info(f"Would backup: {ckpt.experiment_name}/{ckpt.local_path.name} ({ckpt.size_human()})")
                return 0

            # Ensure bucket exists
            self.ensure_bucket_exists()

            # Backup each checkpoint
            success_count = 0
            for checkpoint in pending:
                if self.backup_checkpoint(checkpoint):
                    # Update state
                    if checkpoint.experiment_name not in self.state.backed_up_checkpoints:
                        self.state.backed_up_checkpoints[checkpoint.experiment_name] = {}

                    bucket = self.config["backup"]["destination"]["bucket"]
                    prefix = self.config["backup"]["destination"]["prefix"]
                    s3_path = f"s3://{bucket}/{prefix}/{checkpoint.experiment_name}/{checkpoint.local_path.name}"

                    self.state.backed_up_checkpoints[checkpoint.experiment_name][
                        checkpoint.local_path.name
                    ] = s3_path

                    success_count += 1
                    self._save_state()
                else:
                    self.logger.error(f"Failed to backup checkpoint: {checkpoint.experiment_name}/{checkpoint.local_path.name}")

            self.state.last_run = datetime.now().isoformat()
            self._save_state()

            self.logger.info(
                f"Backup complete: {success_count}/{len(pending)} checkpoint(s) backed up successfully"
            )

            return 0 if success_count == len(pending) else 1

        except EnvironmentError as e:
            self.logger.error(f"Environment configuration error: {e}")
            return 2
        except Exception as e:
            self.logger.exception(f"Unexpected error during backup: {e}")
            return 3
        finally:
            self.release_lock()


def main():
    parser = argparse.ArgumentParser(
        description="Alpha Checkpoint Backup System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Configuration file path (default: config.yaml)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate backup without uploading"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-backup of all checkpoints"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List current backup state and pending checkpoints"
    )
    args = parser.parse_args()

    try:
        manager = CheckpointBackupManager(
            args.config,
            dry_run=args.dry_run,
            force=args.force
        )

        if args.list:
            manager.list_state()
            return 0

        return manager.run()

    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}", file=sys.stderr)
        return 1
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML configuration: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
