import { useCallback, useRef } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useFlow } from "../lib/store";
import { BoxNode } from "./BoxNode";

const nodeTypes = { box: BoxNode };

function CanvasInner() {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addBox,
    catalog,
    setSelected,
  } = useFlow();
  const wrapper = useRef<HTMLDivElement>(null);
  const rf = useReactFlow();

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const boxName = e.dataTransfer.getData("application/x-moire-box");
      const box = catalog.find((b) => b.name === boxName);
      if (!box) return;
      const position = rf.screenToFlowPosition({
        x: e.clientX,
        y: e.clientY,
      });
      addBox(box, position);
    },
    [catalog, rf, addBox]
  );

  return (
    <div
      ref={wrapper}
      onDragOver={onDragOver}
      onDrop={onDrop}
      className="relative h-full w-full"
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onPaneClick={() => setSelected(null)}
        nodeTypes={nodeTypes}
        fitView
        proOptions={{ hideAttribution: true }}
        defaultEdgeOptions={{
          animated: false,
          style: { strokeWidth: 1.5 },
        }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1.2}
          color="rgb(212 212 216)"
        />
        <Controls
          showInteractive={false}
          position="bottom-right"
          className="!shadow-soft"
        />
        <MiniMap
          pannable
          zoomable
          maskColor="rgba(244, 244, 245, 0.7)"
          nodeColor={(n) =>
            n.id === useFlow.getState().selectedNodeId
              ? "rgb(99 102 241)"
              : "rgb(161 161 170)"
          }
          nodeBorderRadius={6}
          style={{ width: 160, height: 100 }}
        />
      </ReactFlow>
    </div>
  );
}

export function Canvas() {
  return (
    <ReactFlowProvider>
      <CanvasInner />
    </ReactFlowProvider>
  );
}
