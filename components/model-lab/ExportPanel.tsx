"use client";

import { useState } from "react";
import { Download, FileJson, Image } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface ExportPanelProps {
  onExportModel: () => void;
  onExportReport: () => void;
  onExportChart: () => Promise<void>;
  modelFormat: "ONNX" | "TF.js";
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
    <div className="flex flex-wrap gap-2">
      <Button variant="outline" size="sm" onClick={onExportModel} className="text-foreground/60 hover:text-foreground/80">
        <Download className="h-3.5 w-3.5" />
        Model
        <Badge variant="secondary" className="ml-1 rounded bg-white/8 px-1.5 py-0.5 text-[10px] font-medium text-foreground/40">
          {modelFormat}
        </Badge>
      </Button>

      <Button variant="outline" size="sm" onClick={onExportReport} className="text-foreground/60 hover:text-foreground/80">
        <FileJson className="h-3.5 w-3.5" />
        Report
        <Badge variant="secondary" className="ml-1 rounded bg-white/8 px-1.5 py-0.5 text-[10px] font-medium text-foreground/40">
          JSON
        </Badge>
      </Button>

      <Button
        variant="outline"
        size="sm"
        onClick={handleChartExport}
        disabled={chartExporting}
        className="text-foreground/60 hover:text-foreground/80"
      >
        <Image className="h-3.5 w-3.5" />
        {chartExporting ? "Exporting..." : "Chart"}
        <Badge variant="secondary" className="ml-1 rounded bg-white/8 px-1.5 py-0.5 text-[10px] font-medium text-foreground/40">
          PNG
        </Badge>
      </Button>
    </div>
  );
}
