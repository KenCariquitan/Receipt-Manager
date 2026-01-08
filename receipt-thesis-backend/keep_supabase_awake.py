"""
Periodic pinger to keep the Supabase project from pausing.

Reads SUPABASE_URL/SUPABASE_ANON_KEY/SUPABASE_DB_URL from .env and
hits a lightweight Auth health endpoint plus a DB `SELECT 1`.

Usage:
  python keep_supabase_awake.py           # loop with default 15m interval
  python keep_supabase_awake.py --once    # single ping then exit
  python keep_supabase_awake.py --interval 600
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from typing import Tuple

import httpx
import psycopg2
from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "").strip()
PROJECT_REF = os.getenv("SUPABASE_PROJECT_REF", "")

DEFAULT_INTERVAL = _env_int("SUPABASE_PING_INTERVAL", 900)  # seconds
DEFAULT_TIMEOUT = _env_int("SUPABASE_PING_TIMEOUT", 10)  # seconds
MIN_INTERVAL = 60


def _build_db_dsn() -> str:
    if not SUPABASE_DB_URL:
        return ""
    if "sslmode=" in SUPABASE_DB_URL:
        return SUPABASE_DB_URL
    sep = "&" if "?" in SUPABASE_DB_URL else "?"
    return f"{SUPABASE_DB_URL}{sep}sslmode=require"


AUTH_HEALTH_URL = f"{SUPABASE_URL}/auth/v1/health" if SUPABASE_URL else ""
DB_DSN = _build_db_dsn()


async def ping_auth(client: httpx.AsyncClient) -> Tuple[bool, str]:
    if not AUTH_HEALTH_URL:
        return False, "SUPABASE_URL not set"
    headers = {"apikey": SUPABASE_ANON_KEY} if SUPABASE_ANON_KEY else {}
    try:
        resp = await client.get(AUTH_HEALTH_URL, headers=headers)
        resp.raise_for_status()
        return True, f"HTTP {resp.status_code}"
    except Exception as exc:
        return False, repr(exc)


def _ping_db_sync() -> Tuple[bool, str]:
    if not DB_DSN:
        return False, "SUPABASE_DB_URL not set"
    try:
        with psycopg2.connect(DB_DSN, connect_timeout=DEFAULT_TIMEOUT) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        return True, "SELECT 1 ok"
    except Exception as exc:
        return False, repr(exc)


async def ping_db() -> Tuple[bool, str]:
    return await asyncio.to_thread(_ping_db_sync)


async def cycle(interval: int, once: bool = False) -> None:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        while True:
            started = time.strftime("%Y-%m-%d %H:%M:%S")
            auth_ok, auth_msg = await ping_auth(client)
            db_ok, db_msg = await ping_db()
            logging.info(
                "[%s] auth:%s (%s) | db:%s (%s)",
                started,
                "OK" if auth_ok else "FAIL",
                auth_msg,
                "OK" if db_ok else "FAIL",
                db_msg,
            )
            if once:
                break
            await asyncio.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep Supabase project awake with periodic pings.")
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Seconds between pings (min {MIN_INTERVAL}, default {DEFAULT_INTERVAL}).",
    )
    parser.add_argument("--once", action="store_true", help="Send a single ping then exit.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    interval = max(MIN_INTERVAL, args.interval)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.info(
        "Starting Supabase keep-alive (project=%s, interval=%ss, timeout=%ss)",
        PROJECT_REF or "unknown",
        interval,
        DEFAULT_TIMEOUT,
    )
    try:
        asyncio.run(cycle(interval, once=args.once))
    except KeyboardInterrupt:
        logging.info("Stopped by user.")


if __name__ == "__main__":
    main()
