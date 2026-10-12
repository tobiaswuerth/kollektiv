import { Component, type ReactNode } from "react";
import { Modal } from "./ui";

/** A failed editor download must leave mission controls and the live graph usable. */
export class EditorBoundary extends Component<
  { close: () => void; children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <Modal title="Unable to open the editor" close={this.props.close}>
        <p>The editor could not be loaded. Reload the page and try again.</p>
        <footer className="action-row">
          <button className="button" onClick={this.props.close}>
            Close
          </button>
          <button className="button primary" onClick={() => location.reload()}>
            Reload page
          </button>
        </footer>
      </Modal>
    );
  }
}
