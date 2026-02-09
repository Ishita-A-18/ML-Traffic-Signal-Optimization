from metrics import MetricsLogger
import subprocess

def run(controller):
    logger = MetricsLogger()
    subprocess.run(["python", controller])
    return logger.results()

fixed = run("fixed_control.py")
ml = run("control.py")

print("FIXED:", fixed)
print("ML:", ml)
