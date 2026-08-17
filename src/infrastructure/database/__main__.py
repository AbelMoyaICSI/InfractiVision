"""CLI de la base de datos local.

Ejecuta la migración de los datos legacy (JSON) a ``data/infractions.sqlite``:

    python -m src.infrastructure.database

Es idempotente: si ya se migró, no vuelve a insertar (usa ``--force`` para
refrescar ``video_configs`` desde los JSON actuales).
"""
from __future__ import annotations

import argparse
import sys

from src.infrastructure.database.app_repository import (
    DEFAULT_DB_PATH,
    AppRepository,
    migrate_legacy_data,
)


def _fmt_bool(v: bool) -> str:
    return "✅" if v else "—"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.infrastructure.database",
        description="Migra los JSON legacy a la base SQLite local.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ejecuta la migración refrescando configs desde los JSON actuales.",
    )
    args = parser.parse_args(argv)

    print("=" * 60)
    print("🗄️  Migración de datos legacy → SQLite")
    print(f"📁 DB: {DEFAULT_DB_PATH}")
    print("=" * 60)

    summary = migrate_legacy_data(force=args.force)

    if summary.get("skipped"):
        print("\n⏭️  La migración ya se ejecutó antes. Nada que insertar.")
        print("   Usa `--force` para refrescar las configuraciones desde los JSON.")

    print("\n📊 Resumen por tabla:")
    for table in ("video_configs", "infractions", "indicators", "migrations"):
        n = summary.get(table, 0)
        print(f"  {table:<16} {n:>6} filas")

    repo = AppRepository(DEFAULT_DB_PATH)
    print("\n🔍 Detalle:")
    print(f"  video_configs: {_fmt_bool(repo.count_rows('video_configs') > 0)} "
          f"({repo.count_rows('video_configs')} videos registrados)")
    print(f"  infractions:   {_fmt_bool(repo.count_rows('infractions') > 0)} "
          f"NID={repo.count_infractions('NID')} | NIE={repo.count_infractions('NIE')}")
    print(f"  indicators:    {_fmt_bool(repo.get_indicators() is not None)}")
    print(f"  migrations:    {_fmt_bool(repo.count_rows('migrations') > 0)} "
          f"({repo.count_rows('migrations')} registros)")

    if args.force and not summary.get("skipped"):
        print("\n⚠️  Con --force, las configuraciones se refrescaron desde los JSON.")

    print("\n✅ Migración completada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())