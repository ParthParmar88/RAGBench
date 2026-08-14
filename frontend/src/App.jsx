import FileUpload from "./components/FileUpload.jsx";
import ChatWindow from "./components/ChatWindow.jsx";

export default function App() {
  return (
    <div className="flex h-dvh flex-col bg-surface text-ink">
      {/* Top bar */}
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-line bg-surface px-5">
        <div className="grid h-7 w-7 place-items-center rounded-lg bg-ink text-sm font-semibold text-white">
          P
        </div>
        <div className="flex items-baseline gap-2.5">
          <h1 className="text-[15px] font-semibold tracking-tight">paraX</h1>
          <span className="hidden text-[13px] text-muted sm:inline">
            Retrieval-Augmented Generation
          </span>
        </div>
      </header>

      {/* Main layout */}
      <div className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-[300px_1fr]">
        <FileUpload />
        <ChatWindow />
      </div>
    </div>
  );
}
