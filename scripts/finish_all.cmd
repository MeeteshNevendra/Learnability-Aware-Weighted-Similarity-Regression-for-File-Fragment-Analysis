@echo off
rem Runs every remaining step in order and stops at the first failure. Safe to start again after an interruption:
rem finished training runs are skipped, interrupted runs resume from runs\<run>\last.pt, and evaluation skips
rem finished outputs. Run from the project directory:  scripts\finish_all.cmd
rem The cost measurement (41_evaluate_local.py --only cost) is not repeated here; it was measured on an idle GPU.
cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe

echo === %date% %time% finish_all: training queue ===>> logs\queue.log
%PY% scripts\31_queue.py components main loto ablations robustness >> logs\queue.log 2>&1
if errorlevel 1 goto fail

echo === %date% %time% finish_all: local evaluation (regression provenance reconstruction loto) ===>> logs\eval_local.log
%PY% scripts\41_evaluate_local.py --only regression provenance reconstruction loto >> logs\eval_local.log 2>&1
if errorlevel 1 goto fail

echo === %date% %time% finish_all: bootstrap local --ari --workers 3 ===>> logs\bootstrap.log
%PY% scripts\60_bootstrap.py local --ari --workers 3 >> logs\bootstrap.log 2>&1
if errorlevel 1 goto fail
echo === %date% %time% finish_all: bootstrap recon --workers 3 ===>> logs\bootstrap.log
%PY% scripts\60_bootstrap.py recon --workers 3 >> logs\bootstrap.log 2>&1
if errorlevel 1 goto fail
%PY% scripts\61_ablation_paired.py > logs\ablation_paired.log 2>&1
if errorlevel 1 goto fail

%PY% scripts\70_tables.py > logs\tables.log 2>&1
if errorlevel 1 goto fail
%PY% scripts\71_figures.py > logs\figures.log 2>&1
if errorlevel 1 goto fail
rem Public copy: the comparison with previously reported values (80_compare_reported.py) is not included.

echo FINISH_ALL DONE %date% %time%> logs\finish_all.status
exit /b 0

:fail
echo FINISH_ALL FAILED %date% %time% (see the last log written)> logs\finish_all.status
exit /b 1
