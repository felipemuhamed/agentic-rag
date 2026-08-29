from pathlib import Path

import pytest

from app.uploads.storage import UnsafeFilenameError, UploadStorage, safe_filename


def test_safe_filename_keeps_a_single_component_and_the_suffix() -> None:
    assert safe_filename("../../etc/manual.MD") == "manual.md"
    assert safe_filename("Manual OLT v2 (rev3).PDF") == "Manual_OLT_v2_rev3.pdf"
    assert safe_filename("  guia.txt  ") == "guia.txt"


def test_safe_filename_rejects_blank_and_traversal_only_names() -> None:
    for value in ("", "   ", "..", "."):
        with pytest.raises(UnsafeFilenameError):
            safe_filename(value)


def test_document_path_is_deterministic_per_tenant(tmp_path: Path) -> None:
    storage = UploadStorage(tmp_path)

    first = storage.document_path(tenant_id="public", filename="guia.md")
    second = storage.document_path(tenant_id="public", filename="guia.md")
    other_tenant = storage.document_path(tenant_id="privado", filename="guia.md")
    anonymous = storage.document_path(tenant_id=None, filename="guia.md")

    assert first == second
    assert first != other_tenant
    assert anonymous.parent.parent.name == "_shared"
    assert storage.contains(first)


def test_tenants_that_share_a_sanitized_name_keep_separate_directories(
    tmp_path: Path,
) -> None:
    storage = UploadStorage(tmp_path)

    first = storage.document_path(tenant_id="tenant/a", filename="guia.md")
    second = storage.document_path(tenant_id="tenant:a", filename="guia.md")

    assert first != second


def test_save_writes_the_file_and_leaves_no_partial_artifact(
    tmp_path: Path,
) -> None:
    storage = UploadStorage(tmp_path)
    path = storage.document_path(tenant_id="public", filename="guia.md")

    storage.save(path, b"conteudo")
    storage.save(path, b"conteudo atualizado")

    assert path.read_bytes() == b"conteudo atualizado"
    assert [item.name for item in path.parent.iterdir()] == ["guia.md"]


def test_remove_ignores_paths_outside_the_storage_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside.md"
    outside.write_text("mantido", encoding="utf-8")
    storage = UploadStorage(tmp_path / "uploads")
    inside = storage.document_path(tenant_id="public", filename="guia.md")
    storage.save(inside, b"conteudo")

    assert storage.remove(outside) is False
    assert storage.remove(inside) is True
    assert storage.remove(inside) is False
    assert outside.exists()
