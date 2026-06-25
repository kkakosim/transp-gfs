REM Compiling and linking with gfortran

@REM prevent conflicts if G95 is installed
set LIBRARY_PATH=

c:\MinGW\bin\gfortran -o calwrf.exe calwrf.f -L. -llibnetcdf-0
