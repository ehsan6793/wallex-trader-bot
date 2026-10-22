@echo off
cd /d %~dp0
echo === %date% %time% > _doctor.txt
where python >> _doctor.txt 2>&1
python --version >> _doctor.txt 2>&1
py --version >> _doctor.txt 2>&1
python -c "import numpy, pandas, httpx; print('libs ok', numpy.__version__, pandas.__version__, httpx.__version__)" >> _doctor.txt 2>&1
echo done >> _doctor.txt
