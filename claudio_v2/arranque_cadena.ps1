# 28/09. Lo corre la tarea programada "CLAUDIO cadena entrenamiento" al iniciar sesion (despues de un
# corte de luz). Relanza cada cadena SOLO si no esta corriendo y no termino. Las cadenas son idempotentes:
# retoman cada entrenamiento desde su ultimo checkpoint (last.pt / checkpoint.pth / last.ckpt).
# Se lanzan via WMI para que el proceso no quede atado a esta tarea (el Programador de tareas mata el arbol
# del proceso cuando la tarea vence) ni a la app de Claude.
# Para sumar una cadena nueva: agregarla a $cadenas con su log y la marca que escribe al terminar.
$dir = 'C:\Users\Tomas\Documents\LAB3\CLAUDIO_AI\claudio_v2'
$cadenas = @(
    # 30/09: cadena_ds26 (RF11 con los planos reales nuevos) corre cadena_ds24 (RF9) al final
    @{ script = 'cadena_ds26.sh'; log = 'logs\log_cadena_ds26.txt'; fin = 'CADENA_DS26_LISTA' },
    # 01/10: RF12 (RF-DETR Small sobre ds27)
    @{ script = 'cadena_ds27.sh'; log = 'logs\log_cadena_ds27.txt'; fin = 'CADENA_DS27_LISTA' },
    # 02/10: RF16 (RF12 Small -> ds28, todos los planos reales); reemplaza a RF15
    @{ script = 'cadena_ds28b.sh'; log = 'logs\log_cadena_ds28b.txt'; fin = 'CADENA_DS28B_LISTA' },
    # 03/10: RF17 = RF16 Small a res 784
    @{ script = 'cadena_ds29.sh'; log = 'logs\log_cadena_ds29.txt'; fin = 'CADENA_DS29_LISTA' }
)
$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'sh.exe' }
# 28/09 (revision): si murio el sh pero quedo un entrenamiento/evaluacion vivo, relanzar la cadena lo duplicaria
$entrenando = Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'py*.exe' -and
    $_.CommandLine -match 'train\.py|train_rfdetr|reanudar\.py|evaluate\.py|build_all\.py|yolo2coco\.py|verif_datos\.py|verif_train\.py|verif_aplicar\.py' }
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
