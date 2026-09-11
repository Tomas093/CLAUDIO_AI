<#
  limpieza.ps1 -- Limpieza del proyecto CLAUDIO_AI

  NO BORRA NADA. Mueve todo lo aprobado a  _A_BORRAR\  conservando la
  estructura de carpetas, para que puedas revisar que quedo adentro y
  eliminar esa carpeta a mano cuando estes tranquilo.

  Uso:
      cd C:\Users\Tomas\Documents\LAB3\CLAUDIO_AI
      powershell -ExecutionPolicy Bypass -File .\limpieza.ps1 -DryRun   # ver que haria
      powershell -ExecutionPolicy Bypass -File .\limpieza.ps1           # hacerlo

  Se conservan explicitamente:
      dxf\            (planos fuente, incluido background.dxf)
      test\           (imagenes de evaluacion)
      zips_datasets\  (unica fuente de datos reales)
      best_models\    -> se RENOMBRA a modelos_v1\
      train-maker\input\components\  y los modificadores reales
      test1.*, test2.*, test_2.dxf   (set de evaluacion)
#>

[CmdletBinding()]
param(
    [switch]$DryRun,
    [string]$Root = $PSScriptRoot
)

if (-not $Root) { $Root = (Get-Location).Path }
Set-Location $Root

$Trash = Join-Path $Root '_A_BORRAR'
$script:moved = 0
$script:skipped = 0
$script:bytes = 0

function Move-ToTrash {
    param([string]$RelPath)

    $src = Join-Path $Root $RelPath
    if (-not (Test-Path -LiteralPath $src)) { $script:skipped++; return }

    $item = Get-Item -LiteralPath $src -Force
    $size = if ($item.PSIsContainer) {
        (Get-ChildItem -LiteralPath $src -Recurse -File -ErrorAction SilentlyContinue |
            Measure-Object -Property Length -Sum).Sum
    } else { $item.Length }
    if (-not $size) { $size = 0 }

    $dst = Join-Path $Trash $RelPath
    $dstParent = Split-Path $dst -Parent

    $mb = [math]::Round($size / 1MB, 1)
    Write-Host ("  {0,-62} {1,8} MB" -f $RelPath, $mb)

    if (-not $DryRun) {
        if (-not (Test-Path -LiteralPath $dstParent)) {
            New-Item -ItemType Directory -Path $dstParent -Force | Out-Null
        }
        if (Test-Path -LiteralPath $dst) {
            $dst = "$dst._$(Get-Random -Maximum 99999)"
        }
        Move-Item -LiteralPath $src -Destination $dst -Force
    }
    $script:moved++
    $script:bytes += $size
}

function Move-Glob {
    param([string]$Dir, [string]$Pattern)
    $full = Join-Path $Root $Dir
    if (-not (Test-Path -LiteralPath $full)) { return }
    Get-ChildItem -LiteralPath $full -Filter $Pattern -Force -ErrorAction SilentlyContinue |
        ForEach-Object {
            $rel = if ($Dir) { Join-Path $Dir $_.Name } else { $_.Name }
            Move-ToTrash $rel
        }
}

Write-Host ''
Write-Host '===============================================================' -ForegroundColor Cyan
if ($DryRun) {
    Write-Host '  SIMULACRO -- no se mueve nada' -ForegroundColor Yellow
} else {
    Write-Host '  Moviendo a _A_BORRAR\ ' -ForegroundColor Cyan
}
Write-Host '===============================================================' -ForegroundColor Cyan

# -- 1. Riesgo activo ------------------------------------------------------
Write-Host "`n[1] Script peligroso (reescribe y trunca el labeler)" -ForegroundColor Yellow
Move-ToTrash 'train-maker\patch_fusion.py'

# -- 2. Codigo muerto o roto -----------------------------------------------
Write-Host "`n[2] Codigo muerto o roto" -ForegroundColor Yellow
@(
    'api.py', 'run.py',
    'train-maker\unified.py', 'train-maker\ib_maker.py',
    'train-maker\validate_dataset.py', 'train-maker\verification_renderer.py',
    'train-maker\verify_distribution.py', 'train-maker\verify_bbox.py',
    'train-maker\extract_photos_form_block.py'
) | ForEach-Object { Move-ToTrash $_ }

# -- 3. Duplicados (reemplazados por run_component.py) ---------------------
Write-Host "`n[3] Scripts duplicados" -ForegroundColor Yellow
@(
    'crop_all.py', 'crop_all_2.py', 'crop_positives.py', 'crop_positives_sbc.py',
    'run_tests.py', 'run_tests2.py', 'loop_crop.ps1',
    'count_blocks.py', 'render_blocks.py',
    'train-maker\run_phase2.py', 'train-maker\run_phase2_imm.py',
    'train-maker\run_phase2_sbc.py', 'train-maker\run_phase4_sbc.py',
    'train-maker\run_retrain_sbc.py',
    'sift_share'
) | ForEach-Object { Move-ToTrash $_ }

# -- 4. Artefactos accidentales y temporales -------------------------------
Write-Host "`n[4] Artefactos y temporales" -ForegroundColor Yellow
@(
    '--out.json', '--out.png', 'Claude Setup.exe',
    '__pycache__', 'train-maker\__pycache__', '.idea',
    'train-maker\test_src.txt', 'train-maker\test_dst.txt',
    'train-maker\dataset_mezclado_interruptor_diferencial.yaml',
    'plano3_con_detecciones.dxf', 'plano3.png', 'plano3.json', 'plano3.dxf',
    'test1_render.png', 'test1_render.json',
    'test_2_render.png', 'test_2_render.json',
    'yolo26n.pt', 'scratch'
) | ForEach-Object { Move-ToTrash $_ }
Move-Glob 'train-maker' 'pipeline_run_*.log'

# -- 5. Outputs regenerables -----------------------------------------------
Write-Host "`n[5] Salidas de inferencia (regenerables)" -ForegroundColor Yellow
Get-ChildItem -LiteralPath $Root -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^(out_|pipeline_out)' } |
    ForEach-Object { Move-ToTrash $_.Name }
@(
    'detecciones_recortes', 'inserts_out', 'crops_falsos_positivos'
) | ForEach-Object { Move-ToTrash $_ }

Write-Host "`n[6] Datasets y sprites generados (se regeneran con el pipeline)" -ForegroundColor Yellow
Get-ChildItem -LiteralPath (Join-Path $Root 'train-maker') -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^(dataset_sintetico_|dataset_real_|negatives_review)' } |
    ForEach-Object { Move-ToTrash (Join-Path 'train-maker' $_.Name) }
Move-Glob 'train-maker' '*.cache'
@('train-maker\output', 'train-maker\verification', 'train-maker\models') |
    ForEach-Object { Move-ToTrash $_ }

Write-Host "`n[7] Runs viejos de YOLO (entrenados con el pipeline con bugs)" -ForegroundColor Yellow
Move-ToTrash 'yolo_workspace'

# -- 8. Carpetas ajenas al pipeline ----------------------------------------
Write-Host "`n[8] Codigo de terceros / no usado por el pipeline" -ForegroundColor Yellow
@('Graph', 'Optuna', 'PHT-DBSCAN', 'Sliding Window', 'YOLO', 'dxf-viewer') |
    ForEach-Object { Move-ToTrash $_ }

# -- 9. Modificadores: dejar solo polos y anotaciones ----------------------
Write-Host "`n[9] input\modifiers -- se conservan solo polos y anotaciones" -ForegroundColor Yellow
$modDir = Join-Path $Root 'train-maker\input\modifiers'
if (Test-Path -LiteralPath $modDir) {
    $keep = 0
    Get-ChildItem -LiteralPath $modDir -File -Force | ForEach-Object {
        if ($_.Name -match '^(mod_|text_mod_)') {
            $keep++
        } else {
            Move-ToTrash (Join-Path 'train-maker\input\modifiers' $_.Name)
        }
    }
    Write-Host "  -> se conservan $keep archivos (mod_*, text_mod_*)" -ForegroundColor Green
}

# -- 10. Renombrar best_models ---------------------------------------------
Write-Host "`n[10] Renombrar best_models -> modelos_v1" -ForegroundColor Yellow
$bm = Join-Path $Root 'best_models'
$v1 = Join-Path $Root 'modelos_v1'
if (Test-Path -LiteralPath $bm) {
    if (Test-Path -LiteralPath $v1) {
        Write-Host '  ya existe modelos_v1\, no se toca' -ForegroundColor DarkYellow
    } else {
        Write-Host '  best_models\ -> modelos_v1\'
        if (-not $DryRun) { Rename-Item -LiteralPath $bm -NewName 'modelos_v1' }
    }
} else {
    Write-Host '  best_models\ no existe' -ForegroundColor DarkGray
}

# -- Resumen ---------------------------------------------------------------
$mbTotal = [math]::Round($script:bytes / 1MB, 1)
Write-Host ''
Write-Host '===============================================================' -ForegroundColor Cyan
Write-Host ("  {0} elementos, {1} MB   ({2} no existian)" -f $script:moved, $mbTotal, $script:skipped)
if ($DryRun) {
    Write-Host '  SIMULACRO: no se movio nada. Saca -DryRun para ejecutar.' -ForegroundColor Yellow
} else {
    Write-Host "  Todo quedo en:  $Trash" -ForegroundColor Green
    Write-Host '  Revisalo y borralo a mano cuando estes seguro.' -ForegroundColor Green
}
Write-Host '===============================================================' -ForegroundColor Cyan
Write-Host ''
Write-Host 'OJO: crops_falsos_positivos\ se movio tambien. Si esos recortes los' -ForegroundColor Yellow
Write-Host 'curaste a mano como falsos positivos reales, rescatalos y ponelos en' -ForegroundColor Yellow
Write-Host 'train-maker\negatives_<componente>\ : el pipeline los usa como hard' -ForegroundColor Yellow
Write-Host 'negatives, y valen mucho mas que los negativos sinteticos.' -ForegroundColor Yellow
Write-Host ''
