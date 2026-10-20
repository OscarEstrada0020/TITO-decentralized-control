"""Decentralized control of a two-input, two-output linear system.

The script compares two decentralized control configurations:

1. Direct pairing:
   y1 is controlled using plant input u1.
   y2 is controlled using plant input u2.

2. Cross pairing:
   y1 is controlled using plant input u2.
   y2 is controlled using plant input u1.

Matrix convention
-----------------
For a MIMO transfer-function matrix G:

    G[i, j] = transfer function from input u_j to output y_i

For the closed-loop transfer-function matrix T:

    T[i, j] = transfer function from reference r_j to output y_i
"""

from pathlib import Path

import control as ctl
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

SIMULATION_END_TIME = 20.0
NUMBER_OF_TIME_POINTS = 2000
MINREAL_TOLERANCE = 1e-6

FIGURES_DIRECTORY = Path("figures")


# ============================================================
# SYSTEM DEFINITION
# ============================================================

def create_plant() -> ctl.TransferFunction:
    """Create the 2x2 plant transfer-function matrix.

    Returns
    -------
    control.TransferFunction
        TITO plant model with the structure

            G = [[G11, G12],
                 [G21, G22]]
    """
    numerator = [
        [[16.9], [36.12]],
        [[-9.57], [-4.17]],
    ]

    denominator = [
        [[1, 5], [1, 11]],
        [[1, 5], [1, 11]],
    ]

    return ctl.tf(
        numerator,
        denominator,
        inputs=["u1", "u2"],
        outputs=["y1", "y2"],
        name="plant",
    )


# ============================================================
# CONTROLLER DEFINITIONS
# ============================================================

def create_direct_pairing_controller() -> ctl.TransferFunction:
    """Create the controller for direct input-output pairing.

    Pairing
    -------
    y1 is controlled using u1.
    y2 is controlled using u2.

    The controller matrix is

        C_direct = [[C11,   0],
                    [  0, C22]]
    """
    numerator = [
        [[0.05, 0.15], [0]],
        [[0], [30, 9]],
    ]

    denominator = [
        [[1, 0], [1]],
        [[1], [1, 30, 0]],
    ]

    return ctl.tf(
        numerator,
        denominator,
        inputs=["e1", "e2"],
        outputs=["u1", "u2"],
        name="direct_pairing_controller",
    )


def create_cross_pairing_controller() -> ctl.TransferFunction:
    """Create the controller for cross input-output pairing.

    Pairing
    -------
    y1 is controlled using u2.
    y2 is controlled using u1.

    The controller matrix is

        C_cross = [[  0, C12],
                   [C21,   0]]
    """
    numerator = [
        [[0], [-1]],
        [[0.8], [0]],
    ]

    denominator = [
        [[1], [1, 0]],
        [[1, 0], [1]],
    ]

    return ctl.tf(
        numerator,
        denominator,
        inputs=["e1", "e2"],
        outputs=["u1", "u2"],
        name="cross_pairing_controller",
    )


# ============================================================
# MULTIVARIABLE STRUCTURE FUNCTIONS
# ============================================================

def calculate_structure_functions(
    plant: ctl.TransferFunction,
) -> tuple[ctl.TransferFunction, ctl.TransferFunction]:
    """Calculate structure functions for both control pairings.

    Parameters
    ----------
    plant : control.TransferFunction
        TITO plant transfer-function matrix.

    Returns
    -------
    gamma_direct : control.TransferFunction
        Structure function associated with direct pairing.

    gamma_cross : control.TransferFunction
        Structure function associated with cross pairing.
    """
    gamma_direct = (
        plant[1, 0] * plant[0, 1]
        / (plant[0, 0] * plant[1, 1])
    )

    gamma_cross = (
        plant[0, 0] * plant[1, 1]
        / (plant[1, 0] * plant[0, 1])
    )

    gamma_direct = ctl.minreal(
        gamma_direct,
        tol=MINREAL_TOLERANCE,
        verbose=False,
    )

    gamma_cross = ctl.minreal(
        gamma_cross,
        tol=MINREAL_TOLERANCE,
        verbose=False,
    )

    return gamma_direct, gamma_cross


# ============================================================
# INDIVIDUAL CHANNEL ANALYSIS
# ============================================================

def analyze_direct_pairing_channels(
    plant: ctl.TransferFunction,
    gamma_direct: ctl.TransferFunction,
) -> dict[str, ctl.TransferFunction]:
    """Calculate the individual channels for direct pairing.

    The direct pairing is

        y1 controlled through u1 using C11
        y2 controlled through u2 using C22

    Parameters
    ----------
    plant : control.TransferFunction
        TITO plant.

    gamma_direct : control.TransferFunction
        Structure function for the direct pairing.

    Returns
    -------
    dict
        Controller, channel, loop, and complementary-sensitivity
        transfer functions for both loops.
    """
    controller_11 = 0.05 * ctl.tf([1, 3], [1, 0])
    controller_22 = 30 * ctl.tf([1, 0.3], [1, 30, 0])

    complementary_loop_2 = (
        controller_22 * plant[1, 1]
        / (1 + controller_22 * plant[1, 1])
    )

    channel_1 = ctl.minreal(
        plant[0, 0] * (1 - gamma_direct * complementary_loop_2),
        tol=MINREAL_TOLERANCE,
        verbose=False,
    )

    loop_1 = controller_11 * channel_1

    complementary_sensitivity_1 = ctl.minreal(
        loop_1 / (1 + loop_1),
        tol=MINREAL_TOLERANCE,
        verbose=False,
    )

    complementary_loop_1 = (
        controller_11 * plant[0, 0]
        / (1 + controller_11 * plant[0, 0])
    )

    channel_2 = ctl.minreal(
        plant[1, 1] * (1 - gamma_direct * complementary_loop_1),
        tol=MINREAL_TOLERANCE,
        verbose=False,
    )

    loop_2 = controller_22 * channel_2

    complementary_sensitivity_2 = ctl.minreal(
        loop_2 / (1 + loop_2),
        tol=MINREAL_TOLERANCE,
        verbose=False,
    )

    return {
        "controller_11": controller_11,
        "controller_22": controller_22,
        "channel_1": channel_1,
        "channel_2": channel_2,
        "loop_1": loop_1,
        "loop_2": loop_2,
        "complementary_sensitivity_1": complementary_sensitivity_1,
        "complementary_sensitivity_2": complementary_sensitivity_2,
    }


def analyze_cross_pairing_channels(
    plant: ctl.TransferFunction,
    gamma_cross: ctl.TransferFunction,
) -> dict[str, ctl.TransferFunction]:
    """Calculate the individual channels for cross pairing.

    The cross pairing is

        y1 controlled through u2 using C21
        y2 controlled through u1 using C12

    Parameters
    ----------
    plant : control.TransferFunction
        TITO plant.

    gamma_cross : control.TransferFunction
        Structure function for the cross pairing.

    Returns
    -------
    dict
        Controller, channel, loop, and complementary-sensitivity
        transfer functions for both loops.
    """
    controller_12 = ctl.tf([-1], [1, 0])
    controller_21 = ctl.tf([0.8], [1, 0])

    complementary_loop_2 = (
        controller_12 * plant[1, 0]
        / (1 + controller_12 * plant[1, 0])
    )

    channel_1 = ctl.minreal(
        plant[0, 1] * (1 - gamma_cross * complementary_loop_2),
        tol=MINREAL_TOLERANCE,
        verbose=False,
    )

    loop_1 = controller_21 * channel_1

    complementary_sensitivity_1 = ctl.minreal(
        loop_1 / (1 + loop_1),
        tol=MINREAL_TOLERANCE,
        verbose=False,
    )

    complementary_loop_1 = (
        controller_21 * plant[0, 1]
        / (1 + controller_21 * plant[0, 1])
    )

    channel_2 = ctl.minreal(
        plant[1, 0] * (1 - gamma_cross * complementary_loop_1),
        tol=MINREAL_TOLERANCE,
        verbose=False,
    )

    loop_2 = controller_12 * channel_2

    complementary_sensitivity_2 = ctl.minreal(
        loop_2 / (1 + loop_2),
        tol=MINREAL_TOLERANCE,
        verbose=False,
    )

    return {
        "controller_12": controller_12,
        "controller_21": controller_21,
        "channel_1": channel_1,
        "channel_2": channel_2,
        "loop_1": loop_1,
        "loop_2": loop_2,
        "complementary_sensitivity_1": complementary_sensitivity_1,
        "complementary_sensitivity_2": complementary_sensitivity_2,
    }


# ============================================================
# CLOSED-LOOP SYSTEM
# ============================================================

def build_closed_loop(
    plant: ctl.TransferFunction,
    controller: ctl.TransferFunction,
) -> ctl.StateSpace:
    """Build the reference-to-output closed-loop system.

    The control architecture is

        r -> error -> C(s) -> G(s) -> y
        ^                              |
        |________ negative feedback ___|

    The loop transfer matrix is

        L(s) = G(s) C(s)

    Parameters
    ----------
    plant : control.TransferFunction
        TITO plant model.

    controller : control.TransferFunction
        2x2 controller transfer-function matrix.

    Returns
    -------
    control.StateSpace
        Closed-loop system from references [r1, r2] to outputs
        [y1, y2].
    """
    plant_ss = ctl.ss(plant)
    controller_ss = ctl.ss(controller)

    # ctl.series(A, B) represents B * A. Therefore, this is G * C.
    loop_system = ctl.series(controller_ss, plant_ss)

    identity_feedback = ctl.ss(
        [],
        [],
        [],
        np.eye(2),
        inputs=["y1", "y2"],
        outputs=["feedback_1", "feedback_2"],
    )

    return ctl.feedback(loop_system, identity_feedback, sign=-1)


# ============================================================
# SYSTEM CHECKS
# ============================================================

def is_asymptotically_stable(system: ctl.StateSpace) -> bool:
    """Return True if every continuous-time pole is in the left half-plane."""
    poles = ctl.poles(system)
    return bool(np.all(np.real(poles) < 0))


def print_closed_loop_summary(
    closed_loop: ctl.StateSpace,
    controller_name: str,
) -> None:
    """Print closed-loop poles, stability, and DC gain."""
    poles = ctl.poles(closed_loop)
    stable = is_asymptotically_stable(closed_loop)
    dc_gain = np.asarray(ctl.dcgain(closed_loop))

    print(f"\n{controller_name}")
    print("=" * len(controller_name))

    print("\nClosed-loop poles:")
    for pole in poles:
        print(f"  {pole:.6g}")

    print(f"\nAsymptotically stable: {stable}")

    print("\nClosed-loop DC-gain matrix:")
    print(dc_gain)

    if not stable:
        print(
            "\nWarning: the closed-loop system contains at least one "
            "pole outside the open left half-plane."
        )


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_tito_step_responses(
    system,
    title: str,
    end_time: float = SIMULATION_END_TIME,
    number_of_points: int = NUMBER_OF_TIME_POINTS,
    save_path: Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot all four step responses of a 2x2 system.

    The plots represent

        [0, 0]: input 1 -> output 1
        [0, 1]: input 2 -> output 1
        [1, 0]: input 1 -> output 2
        [1, 1]: input 2 -> output 2

    Parameters
    ----------
    system
        Two-input, two-output LTI system.

    title : str
        Figure title.

    end_time : float, optional
        Final simulation time.

    number_of_points : int, optional
        Number of samples in the simulation time vector.

    save_path : pathlib.Path or None, optional
        Destination of the saved figure. The figure is not saved when
        this argument is None.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Created figure.

    axes : numpy.ndarray
        Two-dimensional array of subplot axes.
    """
    time = np.linspace(0.0, end_time, number_of_points)

    time_r1, response_r1 = ctl.step_response(
        system,
        T=time,
        input=0,
        squeeze=True,
    )

    time_r2, response_r2 = ctl.step_response(
        system,
        T=time,
        input=1,
        squeeze=True,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7),
        sharex=True,
        constrained_layout=True,
    )

    responses = (
        (axes[0, 0], time_r1, response_r1[0], r"$r_1 \rightarrow y_1$"),
        (axes[0, 1], time_r2, response_r2[0], r"$r_2 \rightarrow y_1$"),
        (axes[1, 0], time_r1, response_r1[1], r"$r_1 \rightarrow y_2$"),
        (axes[1, 1], time_r2, response_r2[1], r"$r_2 \rightarrow y_2$"),
    )

    for axis, response_time, response, subplot_title in responses:
        axis.plot(response_time, response, linewidth=1.8)
        axis.set_title(subplot_title)
        axis.set_xlabel("Time")
        axis.set_ylabel("Output")
        axis.grid(True, alpha=0.6)

    figure.suptitle(title, fontsize=14)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=300, bbox_inches="tight")

    return figure, axes


def plot_loop_analysis(
    loop_system: ctl.TransferFunction,
    loop_name: str,
) -> None:
    """Plot the Nyquist diagram and Bode diagram of one control loop."""
    ctl.nyquist_plot(loop_system)
    plt.title(f"Nyquist diagram: {loop_name}")

    ctl.bode_plot(
        loop_system,
        dB=True,
        Hz=False,
        deg=True,
        grid=True,
        display_margins=True,
        title=f"Bode diagram: {loop_name}",
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main() -> None:
    """Run the complete TITO control analysis."""
    plant = create_plant()

    gamma_direct, gamma_cross = calculate_structure_functions(plant)

    direct_channel_results = analyze_direct_pairing_channels(
        plant,
        gamma_direct,
    )

    cross_channel_results = analyze_cross_pairing_channels(
        plant,
        gamma_cross,
    )

    direct_controller = create_direct_pairing_controller()
    cross_controller = create_cross_pairing_controller()

    direct_closed_loop = build_closed_loop(
        plant,
        direct_controller,
    )

    cross_closed_loop = build_closed_loop(
        plant,
        cross_controller,
    )

    print_closed_loop_summary(
        direct_closed_loop,
        "Direct-pairing controller",
    )

    print_closed_loop_summary(
        cross_closed_loop,
        "Cross-pairing controller",
    )

    # Open-loop plant responses.
    plot_tito_step_responses(
        plant,
        title="Open-loop TITO step responses",
        save_path=FIGURES_DIRECTORY / "open_loop_step_responses.png",
    )

    # Direct-pairing closed-loop responses.
    plot_tito_step_responses(
        direct_closed_loop,
        title="Closed-loop responses: direct pairing",
        save_path=FIGURES_DIRECTORY / "direct_pairing_responses.png",
    )

    # Cross-pairing closed-loop responses.
    plot_tito_step_responses(
        cross_closed_loop,
        title="Closed-loop responses: cross pairing",
        save_path=FIGURES_DIRECTORY / "cross_pairing_responses.png",
    )

    # Optional individual-loop analysis.
    #
    # Uncomment one or more calls as needed:
    #
    # plot_loop_analysis(
    #     direct_channel_results["loop_1"],
    #     "Direct pairing, loop 1",
    # )
    #
    # plot_loop_analysis(
    #     direct_channel_results["loop_2"],
    #     "Direct pairing, loop 2",
    # )
    #
    # plot_loop_analysis(
    #     cross_channel_results["loop_1"],
    #     "Cross pairing, loop 1",
    # )
    #
    # plot_loop_analysis(
    #     cross_channel_results["loop_2"],
    #     "Cross pairing, loop 2",
    # )

    # Optional multivariable structure-function diagrams.
    #
    # ctl.bode_plot(
    #     gamma_direct,
    #     dB=True,
    #     Hz=False,
    #     deg=True,
    #     grid=True,
    # )
    #
    # ctl.nyquist_plot(gamma_cross)

    plt.show()


if __name__ == "__main__":
    main()