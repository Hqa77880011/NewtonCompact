"""Damped nodal Newton solve using conservative learned device stamps."""

import argparse
import json
from pathlib import Path

import numpy as np


def assemble(voltage, problem, device_response):
    count = len(voltage)
    residual, jacobian, charges = np.zeros(count), np.zeros((count, count)), []
    alpha = float(problem.get("backward_euler_alpha", 0.))
    for resistor in problem.get("resistors", []):
        first, second, conductance = resistor
        if conductance <= 0:
            raise ValueError("Resistor conductance must be positive")
        current = conductance * (voltage[first] - voltage[second])
        residual[first] += current
        residual[second] -= current
        jacobian[first, first] += conductance
        jacobian[second, second] += conductance
        jacobian[first, second] -= conductance
        jacobian[second, first] -= conductance
    for first, second, current in problem.get("current_sources", []):
        residual[first] += current
        residual[second] -= current
    for device in problem.get("devices", []):
        nodes = device["nodes"]
        if len(nodes) != 4:
            raise ValueError("Each compact device has three terminals plus a reference terminal")
        relative = voltage[nodes[:3]] - voltage[nodes[3]]
        current, charge, stamp = device_response(relative, device["technology"], alpha)
        current, charge, stamp = np.asarray(current), np.asarray(charge), np.asarray(stamp)
        if current.shape != (4,) or charge.shape != (4,) or stamp.shape != (3, 3):
            raise ValueError("Device response must return four currents/charges and a 3x3 stamp")
        if not np.isfinite(np.r_[current, charge, stamp.ravel()]).all():
            raise ValueError("Nonfinite compact device response")
        previous = np.asarray(device.get("previous_charge", [0.] * 4))
        effective = current + alpha * (charge - previous)
        full = np.zeros((4, 4))
        full[:3, :3] = stamp
        full[:3, 3] = -stamp.sum(1)
        full[3, :3] = -stamp.sum(0)
        full[3, 3] = stamp.sum()
        for i, node in enumerate(nodes):
            residual[node] += effective[i]
            for j, other in enumerate(nodes):
                jacobian[node, other] += full[i, j]
        charges.append(charge.tolist())
    return residual, jacobian, charges


def solve(problem, device_response):
    """Solve a fixed-reference resistor/current-source/compact-device netlist."""
    voltage = np.asarray(problem["initial_voltage"], dtype=float).copy()
    fixed = {int(node): float(value) for node, value in problem["fixed_voltage"].items()}
    if not fixed or not np.isfinite(voltage).all():
        raise ValueError("A finite initial state and a fixed voltage reference are required")
    for node, value in fixed.items():
        voltage[node] = value
    unknown = np.array([i for i in range(len(voltage)) if i not in fixed], dtype=int)
    maximum = int(problem.get("max_iterations", 30))
    tolerance = float(problem.get("residual_tolerance", 1e-7))
    if maximum < 1 or tolerance <= 0 or problem.get("backward_euler_alpha", 0.) < 0:
        raise ValueError("Use positive solve limits and nonnegative backward-Euler alpha")
    reason = "iteration_limit"
    for iteration in range(maximum + 1):
        residual, jacobian, charges = assemble(voltage, problem, device_response)
        norm = float(np.max(np.abs(residual[unknown]))) if len(unknown) else 0.
        if norm <= tolerance:
            reason = "converged"
            break
        if iteration == maximum:
            break
        try:
            delta = np.linalg.solve(jacobian[np.ix_(unknown, unknown)], -residual[unknown])
        except np.linalg.LinAlgError:
            reason = "singular_stamp"
            break
        scale = 1.
        while scale >= 1 / 128:
            candidate = voltage.copy()
            candidate[unknown] += scale * delta
            following, _, _ = assemble(candidate, problem, device_response)
            if np.max(np.abs(following[unknown])) < norm:
                voltage = candidate
                break
            scale /= 2
        else:
            reason = "line_search_failed"
            break
    return dict(converged=reason == "converged", reason=reason, iterations=iteration,
                voltage=voltage.tolist(), kcl_residual_inf=norm, terminal_charge=charges)


def main():
    import torch
    from .method import Model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--netlist", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    problem = json.loads(Path(args.netlist).read_text(encoding="utf-8"))
    saved = torch.load(args.model, map_location="cpu", weights_only=True)
    config = dict(saved["config"], alpha=problem.get("backward_euler_alpha", 0.))
    model = Model(config).eval()
    model.load_state_dict(saved["state"])

    def response(voltage, technology, alpha):
        with torch.enable_grad():
            output = model({"x": torch.tensor(voltage[None], dtype=torch.float32),
                            "technology": torch.tensor([technology], dtype=torch.float32)})
        return tuple(output[key][0].detach().numpy() for key in ("terminal_current", "terminal_charge", "stamp"))

    result = solve(problem, response)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
