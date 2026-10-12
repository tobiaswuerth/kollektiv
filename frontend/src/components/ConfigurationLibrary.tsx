import { Copy, Download, Save, Trash2, Upload } from "lucide-react";
import type { TeamConfiguration } from "../types";

export function ConfigurationLibrary({
  draft,
  configurations,
  busy,
  changeName,
  load,
  save,
  remove,
  exportFile,
  importFile,
}: {
  draft: TeamConfiguration;
  configurations: TeamConfiguration[];
  busy: boolean;
  changeName: (name: string) => void;
  load: (id: string) => void;
  save: (copy?: boolean) => void;
  remove: () => void;
  exportFile: () => void;
  importFile: (file: File) => void;
}) {
  return (
    <div className="configuration-library">
      <label className="configuration-name">
        Configuration name
        <input
          value={draft.name}
          onChange={(event) => changeName(event.target.value)}
          placeholder="My collective"
        />
      </label>
      <label className="configuration-load">
        Load saved
        <select
          aria-label="Load saved configuration"
          value={draft.id || ""}
          onChange={(event) => load(event.target.value)}
        >
          <option value="">Choose configuration…</option>
          {configurations.map((entry) => (
            <option key={entry.id} value={entry.id}>
              {entry.name}
            </option>
          ))}
        </select>
      </label>
      <div className="configuration-actions">
        <button
          type="button"
          className="button"
          disabled={busy}
          onClick={() => save()}
        >
          <Save size={14} />
          Save
        </button>
        <button
          type="button"
          className="icon-button"
          disabled={busy}
          onClick={() => save(true)}
          aria-label="Save configuration as copy"
        >
          <Copy size={15} />
        </button>
        <button
          type="button"
          className="icon-button danger"
          disabled={busy || !draft.id}
          onClick={remove}
          aria-label="Delete saved configuration"
        >
          <Trash2 size={15} />
        </button>
        <span className="editor-toolbar-divider" />
        <button
          type="button"
          className="icon-button"
          onClick={exportFile}
          aria-label="Export configuration JSON"
        >
          <Download size={15} />
        </button>
        <label
          className="icon-button editor-import"
          title="Import configuration JSON"
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              event.currentTarget.querySelector("input")?.click();
            }
          }}
        >
          <Upload size={15} />
          <span className="sr-only">Import configuration JSON</span>
          <input
            type="file"
            accept="application/json,.json"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) importFile(file);
              event.target.value = "";
            }}
          />
        </label>
      </div>
    </div>
  );
}
