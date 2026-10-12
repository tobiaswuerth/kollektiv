import { useEffect, useRef, type ReactNode } from "react";
import { X } from "lucide-react";
import { nice } from "../selectors";
export function Badge({
  status,
  children,
}: {
  status: string;
  children?: ReactNode;
}) {
  return <span className={`badge ${status}`}>{children || nice(status)}</span>;
}
export function Empty({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="empty">
      <strong>{title}</strong>
      <p>{children}</p>
    </div>
  );
}
export function JsonTree({
  value,
  label = "checkpoint",
  open = true,
}: {
  value: unknown;
  label?: string;
  open?: boolean;
}) {
  if (value !== null && typeof value === "object") {
    const entries = Object.entries(value);
    return (
      <details className="json-branch" open={open || undefined}>
        <summary>
          <span>{label}</span>
          <small>
            {Array.isArray(value) ? "[" : "{"}
            {entries.length}
            {Array.isArray(value) ? "]" : "}"}
          </small>
        </summary>
        <div>
          {entries.map(([key, item]) => (
            <JsonTree key={key} label={key} value={item} open={false} />
          ))}
        </div>
      </details>
    );
  }
  return (
    <div className="json-leaf">
      <span>{label}</span>: {JSON.stringify(value)}
    </div>
  );
}
export function Modal({
  title,
  close,
  children,
}: {
  title: string;
  close: () => void;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current!;
    dialog.showModal();
    return () => dialog.close();
  }, []);
  return (
    <dialog
      ref={ref}
      className="modal"
      aria-label={title}
      onCancel={close}
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          const box = e.currentTarget.getBoundingClientRect();
          if (
            e.clientX < box.left ||
            e.clientX > box.right ||
            e.clientY < box.top ||
            e.clientY > box.bottom
          )
            close();
        }
      }}
    >
      <header>
        <div>
          <small>COLLECTIVE CONTROL</small>
          <h2>{title}</h2>
        </div>
        <button
          className="icon-button"
          onClick={close}
          aria-label="Close dialog"
        >
          <X size={18} />
        </button>
      </header>
      {children}
    </dialog>
  );
}
export function download(name: string, content: string) {
  const url = URL.createObjectURL(new Blob([content], { type: "text/plain" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
