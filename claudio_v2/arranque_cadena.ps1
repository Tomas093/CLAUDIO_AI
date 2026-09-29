# 28/09. Lo corre la tarea programada "CLAUDIO cadena entrenamiento" al iniciar sesion (despues de un
# corte de luz). Relanza cada cadena SOLO si no esta corriendo y no termino. Las cadenas son idempotentes:
# retoman cada entrenamiento desde su ultimo checkpoint (last.pt / checkpoint.pth / last.ckpt).
# Se lanzan via WMI para que el proceso no quede atado a esta tarea (el Programador de tareas mata el arbol
# del proceso cuando la tarea vence) ni a la app de Claude.
# Para sumar una cadena nueva: agregarla a $cadenas con su log y la marca que escribe al terminar.
$dir = 'C:\Users\Tomas\Documents\LAB3\CLAUDIO_AI\claudio_v2'
$cadenas = @(
    @{ script = 'cadena_orden.sh'; log = 'log_cadena_orden.txt'; fin = 'CADENA_ORDEN_LISTA' },
    @{ script = 'cadena_ds21.sh';  log = 'log_cadena_ds21.txt';  fin = 'CADENA_DS21_LISTA' }
)
$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'sh.exe' }
# 28/09 (revision): si murio el sh pero quedo un entrenamiento/evaluacion vivo, relanzar la cadena lo duplicaria
$entrenando = Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'py*.exe' -and
    $_.CommandLine -match 'train\.py|train_rfdetr|reanudar\.py|evaluate\.py|build_all\.py|yolo2coco\.py' }
if ($entrenando) { exit 0 }
foreach ($c in $cadenas) {
    if (-not (Test-Path (Join-Path $dir $c.script))) { continue }
    $log = Join-Path $dir $c.log
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern $c.fin -Quiet)) { continue }
    if ($procs | Where-Object { $_.CommandLine -like "*$($c.script)*" }) { continue }
    Add-Content -Path $log -Value "[arranque] $(Get-Date -Format 'ddd MMM dd HH:mm:ss yyyy') relanzada por la tarea programada"
    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = "`"C:\Program Files\Git\usr\bin\sh.exe`" $($c.script)"; CurrentDirectory = $dir } | Out-Null
}
