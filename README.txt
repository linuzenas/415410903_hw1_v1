Computer Vision (4105109) - Assignment #2
Student ID: 415410903
Name: Petlu Linu Zenas Paul

What this is
------------
Python code for the three tasks of the assignment: estimate the projection
matrix from 2D-3D correspondences (normalized DLT), split it into K, R and t,
and measure the reprojection error. The report is in the .docx file.

Files
-----
camera_utils.py                      functions used by all steps (DLT, projection, load/save)
step1_check_data.py                  checks the input data
step2_projection_matrix.py           task 1: projection matrix P
step3_decompose_camera.py            task 2: K, R, t (and camera centre)
step4_check_decomposition.py         checks that K[R|t] gives P back
step5_reprojection_error.py          task 3: reprojection errors and two figures
step6_extra_checks.py                extra: normalization on/off and 5-fold hold-out
cv_assignment_correspondences.csv    the 298 points (columns u, v, X, Y, Z),
                                     copied from the Pt2D and Pt3D lists in the assignment PDF
results/                             files written by the scripts (they are overwritten when you run them again)
  console_output.txt                 what the six scripts printed

How to run (Windows, PowerShell or the VS Code terminal)
--------------------------------------------------------
1. Open the folder that contains these files, for example:
       cd C:\path\to\this\folder
2. Install the packages (once):
       py -m pip install numpy pandas scipy matplotlib
3. Run the scripts in this order:
       py step1_check_data.py
       py step2_projection_matrix.py
       py step3_decompose_camera.py
       py step4_check_decomposition.py
       py step5_reprojection_error.py
       py step6_extra_checks.py
Steps 3 to 5 need the files that the previous step wrote into results/.
The scripts find their files relative to their own location, so they also work
when started from another folder.

The files in results/ were produced with Python 3.12.3, numpy 2.4.4, pandas 3.0.2,
scipy 1.17.1 and matplotlib 3.10.8.
