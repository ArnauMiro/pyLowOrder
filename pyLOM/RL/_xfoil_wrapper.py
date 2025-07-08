import aerosandbox.numpy as np
import subprocess
from pathlib import Path
from typing import Union, List, Dict
import tempfile
import warnings
import os
import re
from aerosandbox import XFoil as XFoilAero

# this class is an adatpation of aerosandbox.XFoil to make it work 
# the installation of xfoil tested is https://github.com/RobotLocomotion/xfoil/
class XFoil(XFoilAero):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def _default_keystrokes(
        self,
        airfoil_filename: str,
        output_filename: str,
    ) -> List[str]:
        """
        Returns a list of XFoil keystrokes that are common to all XFoil runs.

        Returns:
            A list of strings, each of which is a single XFoil keystroke to be followed by <enter>.
        """
        run_file_contents = []

        # Disable graphics
        run_file_contents += [
            "plop",
            "g",
            "w 0.05",
            "",
        ] 

        # Load the airfoil
        run_file_contents += [
            f"load {airfoil_filename}",
            "pane" # I'VE ADDED THIS BECAUSE XFOIL WON'T WORK OTHERWISE
        ]

        if self.xfoil_repanel:
            run_file_contents += [
                "ppar",
                f"n {self.xfoil_repanel_n_points}",
                "",
                "",
                "",
                "pane",
            ]

        # Enter oper mode
        run_file_contents += [
            "oper",
        ]

        # Handle Re
        if self.Re != 0:
            run_file_contents += [
                f"v {self.Re:.8g}",
            ]

        # Handle mach
        run_file_contents += [
            f"m {self.mach:.8g}",
        ]

        # Handle hinge moment
        # run_file_contents += [
        #     "hinc",
        #     f"fnew {float(self.hinge_point_x):.8g} {float(self.airfoil.local_camber(self.hinge_point_x)):.8g}",
        #     "fmom",
        # ] # THIS IS REMOVED FROM THE ORIGINAL CODE

        if self.full_potential:
            run_file_contents += [
                "full",
                "fpar",
                f"i {self.max_iter}",
                "",
            ]

        # Handle iterations
        run_file_contents += [
            f"iter {self.max_iter}",
        ]

        # Handle trips and ncrit
        if not (self.xtr_upper == 1 and self.xtr_lower == 1 and self.n_crit == 9):
            run_file_contents += [
                "vpar",
                f"xtr {self.xtr_upper:.8g} {self.xtr_lower:.8g}",
                f"n {self.n_crit:.8g}",
                "",
            ]

        # Set polar accumulation
        run_file_contents += [
            "pacc",
            f"{output_filename}",
            "",
        ]

        # Include more data in polar
        # run_file_contents += ["cinc"]  # include minimum Cp

        return run_file_contents


    def _run_xfoil(
        self,
        run_command: list, # CHANGED FROM ORIGINAL
        read_bl_data_from: str = None,
    ) -> Dict[str, np.ndarray]:
        """
        Private function to run XFoil.

        Args: run_command: A string with any XFoil keystroke inputs that you'd like. By default, you start off within the OPER
        menu. All of the inputs indicated in the constructor have been set already, but you can override them here (for
        this run only) if you want.

        Returns: A dictionary containing all converged solutions obtained with your inputs.

        """
        # Set up a temporary directory
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)

            ### Alternatively, work in another directory:
            if self.working_directory is not None:
                directory = Path(self.working_directory)  # For debugging

            # Designate an intermediate file for file I/O
            output_filename = "output.txt"

            # Handle the airfoil file
            airfoil_file = "airfoil.dat"
            self.airfoil.write_dat(directory / airfoil_file)

            # Handle the keystroke file
            keystrokes = self._default_keystrokes(
                airfoil_filename=airfoil_file, output_filename=output_filename
            )
            keystrokes += run_command # CHANGED FROM ORIGINAL
            keystrokes += ["pacc", "", "quit"]  # End polar accumulation

            # Remove an old output file, if one exists:
            if os.path.exists(directory / output_filename):
                os.remove(directory / output_filename)
            ### Execute
            try:
                # command = f'{self.xfoil_command} {airfoil_file}' # Old syntax; try this if calls are not working
                proc = subprocess.Popen(
                    self.xfoil_command,
                    cwd=directory,
                    stdin=subprocess.PIPE,
                    stdout=None if self.verbose else subprocess.DEVNULL,
                    stderr=None if self.verbose else subprocess.DEVNULL,
                    text=True,
                    # shell=True,
                    # timeout=self.timeout,
                    # check=True
                )
                outs, errs = proc.communicate(
                    input="\n".join(keystrokes), timeout=self.timeout
                )
                return_code = proc.poll()

            except subprocess.TimeoutExpired:
                proc.kill()
                outs, errs = proc.communicate()

                warnings.warn(
                    "XFoil run timed out!\n"
                    "If this was not expected, try increasing the `timeout` parameter\n"
                    "when you create this AeroSandbox XFoil instance.",
                    stacklevel=2,
                )
            except subprocess.CalledProcessError as e:
                if e.returncode == 11:
                    raise self.XFoilError(
                        "XFoil segmentation-faulted. This is likely because your input airfoil has too many points.\n"
                        "Try repaneling your airfoil with `Airfoil.repanel()` before passing it into XFoil.\n"
                        "For further debugging, turn on the `verbose` flag when creating this AeroSandbox XFoil instance."
                    )
                elif e.returncode == 8 or e.returncode == 136:
                    raise self.XFoilError(
                        "XFoil returned a floating point exception. This is probably because you are trying to start\n"
                        "your analysis at an operating point where the viscous boundary layer can't be initialized based\n"
                        "on the computed inviscid flow. (You're probably hitting a Goldstein singularity.) Try starting\n"
                        "your XFoil run at a less-aggressive (alpha closer to 0, higher Re) operating point."
                    )
                elif e.returncode == 1:
                    raise self.XFoilError(
                        f"Command '{self.xfoil_command}' returned non-zero exit status 1.\n"
                        f"This is likely because AeroSandbox does not see XFoil on PATH with the given command.\n"
                        f"Check the logs (`asb.XFoil(..., verbose=True)`) to verify that this is the case, and if so,\n"
                        f"provide the correct path to the XFoil executable in the asb.XFoil constructor via `xfoil_command=`."
                    )
                else:
                    raise e

            ### Parse the polar
            if os.path.exists(directory / output_filename):
                with open(directory / output_filename) as f:
                    lines = f.readlines()
            else:
                raise self.XFoilError(
                    "It appears XFoil didn't produce an output file, probably because it crashed.\n"
                    "To troubleshoot, try some combination of the following:\n"
                    "\t - In the XFoil constructor, verify that either XFoil is on PATH or that the `xfoil_command` parameter is set.\n"
                    "\t - In the XFoil constructor, run with `verbose=True`.\n"
                    "\t - In the XFoil constructor, set the `working_directory` parameter to a known folder to see the XFoil input and output files.\n"
                    "\t - In the XFoil constructor, set the `timeout` parameter to a large number to see if XFoil is just taking a long time to run.\n"
                    "\t - On Windows, use `XFoil.open_interactive()` to run XFoil interactively in a new window.\n"
                    "\t - Try allowing XFoil to repanel the airfoil by setting `xfoil_repanel=True` in the XFoil constructor.\n"
                )

            try:
                separator_line = None
                for i, line in enumerate(lines):
                    # The first line with at least 30 "-" in it is the separator line.
                    if line.count("-") >= 30:
                        separator_line = i
                        break

                if separator_line is None:
                    raise IndexError

                title_line = lines[i - 1]
                columns = title_line.split()

                data_lines = lines[i + 1 :]

            except IndexError:
                raise self.XFoilError(
                    "XFoil output file is malformed; it doesn't have the expected number of lines.\n"
                    "For debugging, the raw output file from XFoil is printed below:\n"
                    + "\n".join(lines)
                    + "\nTitle line: "
                    + title_line
                    + "\nColumns: "
                    + str(columns)
                )

            def str_to_float(s: str) -> float:
                try:
                    return float(s)
                except ValueError:
                    return np.nan

            output = {
                column: []
                for column in [
                    "alpha",
                    "CL",
                    "CD",
                    "CDp",
                    "CM",
                    "Top_Xtr",
                    "Bot_Xtr",
                    "Top_Itr",
                    "Bot_Itr",
                ]
            }

            for pointno, line in enumerate(data_lines):
                float_pattern = r"-?\d+\.\d+"
                entries = re.findall(float_pattern, line)
                data = [str_to_float(entry) for entry in entries]

                if len(data) == 10 and len(columns) == 8:
                    # This is a monkey-patch for a bug in XFoil v6.99, which causes polar output files to be malformed
                    # when including both Cpmin ("cinc") and hinge moment ("hinc") in the same run.
                    columns = [
                        "alpha",
                        "CL",
                        "CD",
                        "CDp",
                        "CM",
                        "Cpmin",
                        "Xcpmin",
                        "Chinge",
                        "Top_Xtr",
                        "Bot_Xtr",
                    ]

                if not len(data) == len(columns):
                    raise self.XFoilError(
                        "XFoil output file is malformed; the header and data have different numbers of columns.\n"
                        "In previous testing, this occurs due to a bug in XFoil itself, with certain input combos.\n"
                        "For debugging, the raw output file from XFoil is printed below:\n"
                        + "\n".join(lines)
                        + "\nTitle line: "
                        + title_line
                        + f"\nIdentified {len(data)} data columns and {len(columns)} header columns."
                        + "\nColumns: "
                        + str(columns)
                        + "\nData: "
                        + str(data)
                    )

                for i in range(len(columns)):
                    output[columns[i]].append(data[i])

            output = {k: np.array(v, dtype=float) for k, v in output.items()}

            return output

    def alpha(
        self,
        alpha: Union[float, np.ndarray],
        start_at: Union[float, None] = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Execute XFoil at a given angle of attack, or at a sequence of angles of attack.

        Args:

            alpha: The angle of attack [degrees]. Can be either a float or an iterable of floats, such as an array.

            start_at: Chooses whether to split a large sweep into two runs that diverge away from some central value,
            to improve convergence. As an example, if you wanted to sweep from alpha=-20 to alpha=20, you might want
            to instead do two sweeps and stitch them together: 0 to 20, and 0 to -20. `start_at` can be either:

                * None, in which case the alpha inputs are run as a single sequence in the order given.

                * A float that corresponds to an angle of attack (in degrees), in which case the alpha inputs are
                split into two sequences that diverge from the `start_at` value. Successful runs are then sorted by
                `alpha` before returning.

        Returns: A dictionary with the XFoil results. Dictionary values are arrays; they may not be the same shape as
        your input array if some points did not converge.

        """
        alphas = np.reshape(np.array(alpha), -1)
        alphas = np.sort(alphas)

        commands = []

        def schedule_run(alpha: float):

            commands.append(f"a {alpha}")

            if self.hinge_point_x is not None:
                commands.append("fmom")

            if self.include_bl_data:
                commands.extend(
                    [
                        f"dump dump_a_{alpha:.8f}.txt",
                        # "vplo",
                        # "cd", # Dissipation coefficient
                        # f"dump cdis_a_{alpha:.8f}.txt",
                        # f"n", # Amplification ratio
                        # f"dump n_a_{alpha:.8f}.txt",
                        # "",
                    ]
                )

        if (
            len(alphas) > 1
            and (start_at is not None)
            and (np.min(alphas) < start_at < np.max(alphas))
        ):
            alphas_upper = alphas[alphas > start_at]
            alphas_lower = alphas[alpha <= start_at][::-1]

            for a in alphas_upper:
                schedule_run(a)

            commands.append("init")

            for a in alphas_lower:
                schedule_run(a)
        else:
            for a in alphas:
                schedule_run(a)
        # print(commands)
        output = self._run_xfoil(
            commands, # CHANGED FROM ORIGINAL
            read_bl_data_from="alpha" if self.include_bl_data else None,
        )
        # I DON'T CARE ABOUT THE SORT ORDER
        # sort_order = np.argsort(output["alpha"])
        # output = {k: v[sort_order] for k, v in output.items()}
        return output


