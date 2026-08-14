import { useEffect, useState } from "react";
import { deleteDocument, listDocuments, uploadPdf } from "../api.js";

function UploadIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 16V4m0 0L7 9m5-5l5 5M5 20h14"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function DocIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      className="shrink-0 text-muted"
      aria-hidden="true"
    >
      <path
        d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path d="M14 3v5h5" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
    </svg>
  );
}

export default function FileUpload({ onChange }) {
  const [docs, setDocs] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function refresh() {
    try {
      setDocs(await listDocuments());
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function handleUpload(e) {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-uploading the same file
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      await uploadPdf(file);
      await refresh();
      onChange?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(title) {
    setBusy(true);
    try {
      await deleteDocument(title);
      await refresh();
      onChange?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <aside className="flex min-h-0 flex-col border-r border-line bg-surface-2">
      <div className="flex items-center justify-between px-5 pb-3 pt-5">
        <span className="eyebrow">Documents</span>
        <span className="text-xs text-muted">{docs.length}</span>
      </div>

      <div className="px-4">
        <label
          className={`btn btn-outline w-full cursor-pointer ${
            busy ? "pointer-events-none opacity-60" : ""
          }`}
        >
          <UploadIcon />
          {busy ? "Working…" : "Upload PDF"}
          <input
            type="file"
            accept="application/pdf"
            onChange={handleUpload}
            disabled={busy}
            hidden
          />
        </label>
      </div>

      {error && (
        <p className="mx-4 mt-3 rounded-lg border border-line bg-surface px-3 py-2 text-xs text-danger">
          {error}
        </p>
      )}

      <div className="mt-4 min-h-0 flex-1 overflow-y-auto px-4 pb-5">
        {docs.length === 0 ? (
          <div className="mt-6 px-2 text-center">
            <p className="text-sm text-secondary">No documents yet</p>
            <p className="mt-1 text-xs text-muted">
              Upload a PDF to start asking questions.
            </p>
          </div>
        ) : (
          <ul className="flex flex-col gap-1.5">
            {docs.map((d) => (
              <li
                key={d.title}
                className="group flex items-center gap-2.5 rounded-lg border border-transparent px-2.5 py-2 transition-colors hover:border-line hover:bg-surface"
              >
                <DocIcon />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[13px] font-medium text-ink" title={d.title}>
                    {d.title}
                  </p>
                  <p className="text-xs text-muted">{d.chunks} chunks</p>
                </div>
                <button
                  className="btn btn-ghost rounded-md px-2 py-1 text-xs text-muted opacity-0 transition-opacity hover:text-danger group-hover:opacity-100"
                  onClick={() => handleDelete(d.title)}
                  disabled={busy}
                  aria-label={`Delete ${d.title}`}
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}
