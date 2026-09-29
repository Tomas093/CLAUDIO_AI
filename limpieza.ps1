<#
  limpieza.ps1 -- Limpieza segura del proyecto CLAUDIO_AI

  NO BORRA NADA PERMANENTEMENTE.
  Mueve todo lo obsoleto a la carpeta de cuarentena `_para_borrar\`
  conservando la estructura de carpetas, de acuerdo con la norma de CLAUDE.md:
  "Borrado: nada de borrar definitivo; mover a _para_borrar".

  Uso:
      powershell -ExecutionPolicy Bypass -File .\limpieza.ps1 -DryRun   # Ver que se moveria
      powershell -ExecutionPolicy Bypass -File .\limpieza.ps1           # Ejecutar el movimiento

  Se conservan explicitamente:
      - claudio_v2\                       (TODO el trabajo activo y su carpeta work\)
      - dxf\                              (Planos fuente, background.dxf, externos, ground truths CSV)
      - test\                             (Ground truth oficial: test_1, test_2)
      - zips_datasets\                    (Fuente primaria de datasets reales de clases)
      - Base de simbolols\                (Fuente de libreria de simbolos electricos)
      - Claude outputs\                   (Grillas numeradas de referencia para revision de Tomas)
      - modelos_v1\                       (Modelos M1 de referencia por componente)
      - test1.dxf, test_2.dxf, TSSS_2 (1).dxf, UNIFILAR TABLERO GENERAL Vyre 09 09 2026.dxf
      - yolo11n.pt                        (Pesos base oficiales de YOLO11 nano para entrenamiento)
      - scale_analyzer.py, run_e2e_tests.py, tools\, tests\, detector_pack\
      - Documentacion y configs (README.md, CLAUDE.md, PROJECT.md, etc.)
#>

[CmdletBinding()]
param(
    [switch]$DryRun,
    [string]$Root = $PSScriptRoot
)

if (-not $Root) { $Root = (Get-Location).Path }
Set-Location $Root

$Trash = Join-Path $Root '_para_borrar'
$script:moved = 0
$script:skipped = 0
$script:bytes = 0

function Move-ToTrash {
    param([string]$RelPath)

    $src = Join-Path $Root $RelPath
    if (-not (Test-Path -LiteralPath $src)) {
        $script:skipped++
        return
    }

    # Proteccion explicita contra tocar claudio_v2
    if ($RelPath -match '^claudio_v2') {
        Write-Warning "SEGURIDAD: Intento bloqueado de tocar $RelPath"
        return
    }

    $item = Get-Item -LiteralPath $src -Force
    $size = if ($item.PSIsContainer) {
        (Get-ChildItem -LiteralPath $src -Recurse -File -Force -ErrorAction SilentlyContinue |
            Measure-Object -Property Length -Sum).Sum
    } else { $item.Length }
    if (-not $size) { $size = 0 }

    $dst = Join-Path $Trash $RelPath
    $dstParent = Split-Path $dst -Parent

    $mb = [math]::Round($size / 1MB, 2)
    Write-Host ("  {0,-65} {1,8} MB" -f $RelPath, $mb)

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
    $full = if ($Dir) { Join-Path $Root $Dir } else { $Root }
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
    Write-Host '  SIMULACRO (DryRun) -- no se mueve ningun archivo' -ForegroundColor Yellow
} else {
    Write-Host "  Moviendo elementos obsoletos a: $Trash" -ForegroundColor Cyan
}
Write-Host '===============================================================' -ForegroundColor Cyan

# -- 1. Runs viejos de YOLO (anteriores a claudio_v2) -----------------------
Write-Host "`n[1] Runs viejos de YOLO (workspace de pipelines anteriores)" -ForegroundColor Yellow
Move-ToTrash 'yolo_workspace'

# -- 2. Bibliotecas de simbolos previas ------------------------------------
Write-Host "`n[2] Bibliotecas de simbolos anteriores (reemplazadas por claudio_v2/data/sym_lib_full)" -ForegroundColor Yellow
@(
    'simbolos_cad_limpios',
    'simbolos_unifilares_limpios',
    'simbolos_from_scratch',
    'simbolos_limpios_universal'
) | ForEach-Object { Move-ToTrash $_ }

# -- 3. Salidas de pipeline obsoletas --------------------------------------
Write-Host "`n[3] Salidas de inferencia de pipelines viejos" -ForegroundColor Yellow
Get-ChildItem -LiteralPath $Root -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^pipeline_out' } |
    ForEach-Object { Move-ToTrash $_.Name }

# -- 4. Salidas de evaluaciones anteriores ---------------------------------
Write-Host "`n[4] Carpetas de evaluación anteriores" -ForegroundColor Yellow
@(
    'eval_multitest', 'output_eval', 'eval_comparativa_con_texto',
    'resultados_suite_eval', 'eval_con_texto', 'eval_comparativa_dual',
    'audit_new_scratch_model', 'resultados_plano_desafio', 'resultados_scratch_eval',
    'output_eval_nuevos', 'evaluation_tsss_2', 'eval_check_fl02',
    'eval_check_fl02_comp', 'evaluation_fl_un_02', 'validation_output',
    'eval_check_tsss2', 'eval_debug_tsss2', 'vis_gen', 'eval_check_test2',
    'evaluation_test2', 'eval_vyre', 'eval_check_test1', 'eval_test_tsss2_clean',
    'eval_test_tsss2', 'evaluation_test1', 'output_eval_benchmarks',
    'eval_vyre_scale120', 'eval_vyre_boosted', 'eval_check_vyre',
    'eval_plano3', 'eval_tsbe', 'eval_tsbe_conf50', 'debug_fps_test1',
    'debug_fps_test2', 'test1_multi_out', 'test2_multi_out', 'resultados',
    'imagenes_detecciones'
) | ForEach-Object { Move-ToTrash $_ }

# -- 5. Visores HTML pesados e incrustados ---------------------------------
Write-Host "`n[5] Visores HTML autonomos pesados" -ForegroundColor Yellow
@(
    'visor_detecciones_corregido.html',
    'visor_planos_industriales.html',
    'visor_planos_completos.html',
    'visor_detecciones.html'
) | ForEach-Object { Move-ToTrash $_ }

# -- 6. Recortes, diagnósticos y temporales en raíz -----------------------
Write-Host "`n[6] Recortes de diagnostico, crops e imagenes sueltas en raiz" -ForegroundColor Yellow
Move-Glob '' 'crop_*.png'
Move-Glob '' 'debug_*.png'
Move-Glob '' 'check_*.png'
Move-Glob '' 'vyre_crop_*.png'
@(
    'c40e_crop.png',
    'scratch_test_crop.png',
    'fl02_bottom_borne.png',
    'fl02_bottom_borne_with_margin.png',
    'f1_coil.png',
    'f2_coil.png',
    'im01_context.png',
    'im01_crop.png',
    'tsss_2_borne.png',
    'temp_debug_tile.jpg',
    'temp_real_itm_tile.jpg',
    'temp_sprite_test.jpg',
    'test_borne_sprite.png',
    'test_draw.png',
    'test_fixed_size.png',
    'test_size.png',
    'UNIFILAR_TABLERO_GENERAL_Vyre_detecciones_completo.png',
    'UNIFILAR_TABLERO_GENERAL_Vyre_diagrama_completo.png',
    'vyre_new_components_visual.png',
    'test_out.txt',
    'test_out2.txt',
    'test_results.txt',
    'full_validation_results.txt',
    'test_probe.dxf',
    'test1.png',
    'test1.json',
    'test2.png',
    'test2.json',
    'test_5_models.py',
    'check_distortions.py',
    'debug_test1.py',
    'fix.py',
    'inspect_fps.py',
    'work'
) | ForEach-Object { Move-ToTrash $_ }

# -- 7. Scripts antiguos / duplicados de la raíz ---------------------------
Write-Host "`n[7] Scripts de pipeline, entrenamientos anteriores y paquetes duplicados en raiz" -ForegroundColor Yellow
@(
    'evaluate_all_plans_backup.py',
    'evaluate_all_plans_old.py',
    'evaluate_all_plans_updated.py',
    'evaluate_all_plans.py',
    'compare_models.py',
    'eval_tsss_2.py',
    'evaluate_on_dxf.py',
    'vector_inference.py',
    'inference_sahi.py',
    'pipeline.py',
    'train_boosted_yolo.py',
    'train_finetune.py',
    'train_scratch_v2.py',
    'train_unified_yolo.py',
    'curate_and_augment_dataset.py',
    'curate_dataset_v3.py',
    'data_centric_curator.py',
    'inject_zero_fn_boost.py',
    'extract_inserts.py',
    'slice_and_filter.py',
    'run_custom_task.py',
    'test_componente_nano.py',
    'dxf_to_image.py',
    'yolo26n.pt',
    'ErrorReports',
    'output_eval',
    'detector_unifilar_pack.zip',
    'Bornera.v2i.yolov11.zip',
    'Diferencial.v3i.yolov11.zip',
    'Termomagnetica.v2i.yolov11.zip'
) | ForEach-Object { Move-ToTrash $_ }

# -- 8. Temporales en dxf/ -------------------------------------------------
Write-Host "`n[8] Archivos temporales o accidentales en dxf\" -ForegroundColor Yellow
@(
    'dxf\ErrorReports',
    'dxf\best.pt',
    'dxf\jijiji.dxf',
    'dxf\jijiji.png',
    'dxf\jijiji.json',
    'dxf\dxf_to_titiles.py',
    'dxf\filter.py'
) | ForEach-Object { Move-ToTrash $_ }

# -- 9. Archivos duplicados en zips_datasets/ -----------------------------
Write-Host "`n[9] Archivos duplicados en zips_datasets\" -ForegroundColor Yellow
@(
    'zips_datasets\Instrumento_de_medicion_multifun.yolov11.zip'
) | ForEach-Object { Move-ToTrash $_ }

# -- 10. Bytecode residual en raiz -----------------------------------------
Write-Host "`n[10] Limpieza de bytecode residual en raiz (__pycache__)" -ForegroundColor Yellow
if (Test-Path (Join-Path $Root '__pycache__')) {
    if (-not $DryRun) {
        Remove-Item -LiteralPath (Join-Path $Root '__pycache__') -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Host "  __pycache__ eliminado de raiz"
}

# -- Resumen ---------------------------------------------------------------
$mbTotal = [math]::Round($script:bytes / 1MB, 2)
Write-Host ''
Write-Host '===============================================================' -ForegroundColor Cyan
Write-Host ("  {0} elementos movidos, {1} MB liberados   ({2} no existian)" -f $script:moved, $mbTotal, $script:skipped)
if ($DryRun) {
    Write-Host '  SIMULACRO: no se movio nada. Ejecute sin -DryRun para aplicar.' -ForegroundColor Yellow
} else {
    Write-Host "  Todo quedo seguro en:  $Trash" -ForegroundColor Green
    Write-Host '  Puede revisarlo y eliminarlo definitivamente a mano cuando desee.' -ForegroundColor Green
}
Write-Host '===============================================================' -ForegroundColor Cyan
Write-Host ''
