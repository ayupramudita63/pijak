from prometheus_client import start_http_server, Gauge
import psutil
import time

# Metrik sistem
cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percent')
memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percent')
disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percent')

def collect_metrics():
    cpu_usage.set(psutil.cpu_percent(interval=1))
    memory_usage.set(psutil.virtual_memory().percent)
    disk_usage.set(psutil.disk_usage('/').percent)

if __name__ == '__main__':
    start_http_server(8000)
    while True:
        collect_metrics()
        time.sleep(15)