"use client";

import { useState } from "react";
import { Download, FileJson, Image as ImageIcon } from "lucide-react";

interface ExportPanelProps {
  onExportModel: () => void;
  onExportReport: () => void;
  onExportChart: () => Promise<void>;
  modelFormat: "ONNX" | "TF.js";
}

function ExportButton({
  icon,
  label,
  format,
  onClick,
  disabled,
}: {
  icon: React.ReactNode;
  label: string;
  format: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button type="button" onClick={onClick} disabled={disabled} className="btn-ghost">
      {icon}
      {label}
      <span className="font-mono text-[11px] text-ink-3">{format}</span>
    </button>
  );
}

export function ExportPanel({
  onExportModel,
  onExportReport,
  onExportChart,
  modelFormat,
}: ExportPanelProps) {
  const [chartExporting, setChartExporting] = useState(false);

  const handleChartExport = async () => {
    setChartExporting(true);
    try {
      await onExportChart();
    } finally {
      setChartExporting(false);
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-2 border-y border-rule py-3">
      <span className="eyebrow mr-2">EXPORT</span>
      <ExportButton icon={<Download className="size-3.5" />} label="Model" format={modelFormat} onClick={onExportModel} />
      <ExportButton icon={<FileJson className="size-3.5" />} label="Report" format="JSON" onClick={onExportReport} />
      <ExportButton
        icon={<ImageIcon className="size-3.5" />}
        label={chartExporting ? "Exporting…" : "Chart"}
        format="PNG"
        onClick={handleChartExport}
        disabled={chartExporting}
      />
    </div>
  );
}
