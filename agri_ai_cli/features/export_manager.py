"""
Sprint 5 - Export Manager.
Export analysis results to JSON, CSV, and TXT report formats.
"""

import json
import csv
import os
from datetime import datetime


class ExportManager:
    """Export farm analysis data to multiple formats."""

    def __init__(self, output_dir="agri_ai_cli/data/exports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_json(self, data, filename=None):
        """Export data as JSON file."""
        if not filename:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return filepath

    def export_csv(self, headers, rows, filename=None):
        """Export tabular data as CSV file."""
        if not filename:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        return filepath

    def export_txt(self, content, filename=None, title="AgriAI Report"):
        """Export analysis as formatted text report."""
        if not filename:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w") as f:
            f.write("=" * 70 + "\n")
            f.write(f"  {title}\n")
            f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  AgriAI CLI v3.0\n")
            f.write("=" * 70 + "\n\n")

            if isinstance(content, dict):
                self._write_dict(f, content)
            elif isinstance(content, str):
                f.write(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        self._write_dict(f, item)
                        f.write("\n" + "-" * 40 + "\n\n")
                    else:
                        f.write(str(item) + "\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("  End of Report\n")
            f.write("=" * 70 + "\n")

        return filepath

    def _write_dict(self, f, d, indent=0):
        """Recursively write dictionary to text file."""
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                f.write(f"{prefix}{key}:\n")
                self._write_dict(f, value, indent + 1)
            elif isinstance(value, list):
                f.write(f"{prefix}{key}:\n")
                for item in value:
                    if isinstance(item, dict):
                        self._write_dict(f, item, indent + 1)
                        f.write(f"{prefix}  ---\n")
                    else:
                        f.write(f"{prefix}  - {item}\n")
            else:
                f.write(f"{prefix}{key}: {value}\n")

    def export_full_report(self, analyses, filename=None):
        """Export comprehensive farm report combining multiple analyses."""
        if not filename:
            filename = f"full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # JSON version
        json_path = self.export_json(analyses, f"{filename}.json")

        # TXT version
        txt_path = self.export_txt(analyses, f"{filename}.txt", "AgriAI Full Farm Report")

        # CSV summary
        csv_rows = []
        csv_headers = ["Analysis", "Key Metric", "Value"]
        if isinstance(analyses, dict):
            for analysis_name, result in analyses.items():
                if isinstance(result, dict):
                    for key, val in result.items():
                        if not isinstance(val, (dict, list)):
                            csv_rows.append([analysis_name, key, str(val)])
        csv_path = self.export_csv(csv_headers, csv_rows, f"{filename}.csv")

        return {
            "json_path": json_path,
            "txt_path": txt_path,
            "csv_path": csv_path,
            "files_created": 3,
        }

    def list_exports(self):
        """List all exported files."""
        if not os.path.exists(self.output_dir):
            return []
        files = []
        for f in sorted(os.listdir(self.output_dir)):
            filepath = os.path.join(self.output_dir, f)
            size = os.path.getsize(filepath)
            files.append({"filename": f, "size_bytes": size, "path": filepath})
        return files

    def get_model_info(self):
        return {
            "name": "Export Manager",
            "formats": ["JSON", "CSV", "TXT"],
            "output_dir": self.output_dir,
            "exports_count": len(self.list_exports()),
        }
