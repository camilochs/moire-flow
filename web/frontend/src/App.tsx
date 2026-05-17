import { TopBar } from "./components/TopBar";
import { Sidebar } from "./components/Sidebar";
import { Canvas } from "./components/Canvas";
import { Inspector } from "./components/Inspector";

export default function App() {
  return (
    <div className="flex h-screen w-screen flex-col bg-canvas dark:bg-canvas-dark">
      <TopBar />
      <div className="flex flex-1 min-h-0">
        <Sidebar />
        <main className="flex-1 min-w-0 relative">
          <Canvas />
        </main>
        <Inspector />
      </div>
    </div>
  );
}
