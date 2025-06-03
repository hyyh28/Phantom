
import os
import json
import datetime
from typing import Dict, Any, Optional, List


class LLMLogger:
    """
    Logger for recording information during the LLM control agent execution process.

    Recorded content includes:
    - Current step information
    - Input context
    - LLM output response
    - Parsed action
    - Error information (if any)
    """

    def __init__(self, log_dir: str = "logs", log_filename: Optional[str] = None, console_output: bool = True):
        """
        Initialize the logger

        Parameters:
            log_dir: Directory to save log files
            log_filename: Log filename, if None a timestamp will be automatically generated
            console_output: Whether to also output logs to console
        """
        self.console_output = console_output

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # If filename not specified, create one using timestamp
        if log_filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"llm_run_{timestamp}.jsonl"

        self.log_path = os.path.join(log_dir, log_filename)

        # Store all log entries for the current session
        self.log_entries = []

        print(f"Logs will be saved to: {self.log_path}")

    def log_step(self, step: int, context: Dict[str, Any],
                 llm_response: Optional[str] = None, action: Optional[Any] = None,
                 error: Optional[str] = None) -> None:
        """
        Record information for a single step

        Parameters:
            step: Current step number
            context: Current context information
            llm_response: LLM response
            action: Parsed action
            error: Error information (if any)
        """
        # Create log entry
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "step": step,
            "context": self._sanitize_for_json(context),
        }

        # Add optional fields
        if llm_response is not None:
            entry["llm_response"] = llm_response

        if action is not None:
            entry["action"] = self._sanitize_for_json(action)

        if error is not None:
            entry["error"] = error

        # Add to in-memory log list
        self.log_entries.append(entry)

        # Write to file
        self._write_to_file(entry)

        # Console output
        if self.console_output:
            self._print_to_console(entry)

    def _sanitize_for_json(self, obj: Any) -> Any:
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # For other types, try to convert to string
            try:
                if hasattr(obj, 'tolist'):  # Handle numpy arrays
                    return obj.tolist()
                else:
                    return str(obj)
            except:
                return f"<Non-serializable object: {type(obj).__name__}>"

    def _write_to_file(self, entry: Dict[str, Any]) -> None:
        """Write log entry to file"""
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def _print_to_console(self, entry: Dict[str, Any]) -> None:
        """Print log entry to console"""
        print(f"\n===== Step {entry['step']} =====")
        print(f"Timestamp: {entry['timestamp']}")

        # Print key context information
        if 'context' in entry:
            context = entry['context']
            print("\nContext information:")
            if 'observation' in context:
                print(f"  Observation: {context['observation']}")
            if 'agent_state' in context:
                print(f"  Agent state: {context['agent_state']}")

        # Print LLM response (may be long, only print first 200 characters)
        if 'llm_response' in entry:
            resp = entry['llm_response']
            print(f"LLM Response: {resp[:200]}..." if len(resp) > 200 else f"LLM Response: {resp}")

        # Print action
        if 'action' in entry:
            print(f"Action: {entry['action']}")

        # Print error
        if 'error' in entry:
            print(f"Error: {entry['error']}")

    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all log entries"""
        return self.log_entries

    def save_summary(self, summary_path: Optional[str] = None) -> None:
        """
        Save log summary

        Parameters:
            summary_path: Summary file path, if None use log file path with .summary suffix
        """
        if summary_path is None:
            summary_path = f"{self.log_path}.summary.json"

        summary = {
            "total_steps": len(self.log_entries),
            "start_time": self.log_entries[0]["timestamp"] if self.log_entries else None,
            "end_time": self.log_entries[-1]["timestamp"] if self.log_entries else None,
            "success_rate": sum(1 for entry in self.log_entries if 'error' not in entry) / len(self.log_entries) if self.log_entries else 0,
            "error_count": sum(1 for entry in self.log_entries if 'error' in entry)
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        if self.console_output:
            print(f"\n===== Log Summary =====")
            print(f"Total steps: {summary['total_steps']}")
            print(f"Start time: {summary['start_time']}")
            print(f"End time: {summary['end_time']}")
            print(f"Success rate: {summary['success_rate']:.2%}")
            print(f"Error count: {summary['error_count']}")
            print(f"Summary saved to: {summary_path}")