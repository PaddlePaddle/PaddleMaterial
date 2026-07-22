"""
本地 TDB 数据库注册表与选择辅助函数
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


TDB_REGISTRY: Dict[str, Dict] = {
    'CoCrFeNiV.TDB-R3.txt': {
        'system_name': 'Co-Cr-Fe-Ni-V 高熵合金数据库',
        'database_kind': 'full_thermodynamic',
        'pycalphad_loadable': True,
        'supported_elements': {'CO', 'CR', 'FE', 'NI', 'V'},
        'recommended_phases': ['FCC_A1', 'BCC_A2', 'HCP_A3', 'SIGMA', 'B2_BCC', 'L12_FCC'],
        'recommended_use': '用于 Co-Cr-Fe-Ni-V 体系的真实 CALPHAD 相平衡计算。',
        'notes': '来源于 Co-Cr-Fe-Ni-V 高熵合金热力学建模；已用等原子 CoCrFeNiV 在 1273.15 K 下完成相平衡验证。',
    },
    'bcc-fcc-Co-Ni-Ti.TDB': {
        'system_name': 'Co-Ni-Ti 命名的 BCC/FCC 简化数据库',
        'database_kind': 'simplified_bcc_fcc',
        'pycalphad_loadable': True,
        'supported_elements': {
            'AL', 'AU', 'CO', 'CR', 'CU', 'FE', 'GA', 'GE', 'HF', 'IR',
            'MO', 'NB', 'NI', 'OS', 'PD', 'PT', 'RE', 'RH', 'RU', 'SB',
            'SC', 'SN', 'TA', 'TC', 'TI', 'V', 'ZN', 'ZR'
        },
        'recommended_phases': ['FCC_A1', 'BCC_A2'],
        'recommended_use': '仅适合 BCC/FCC 两相的简化筛选，不建议当作完整 Co-Ni-Ti 热力学数据库使用。',
        'notes': '文件只有 254 行、2 个相、90 条参数，缺少完整多相/金属间化合物描述。',
    },
    'mc_fe_v2062.tdb': {
        'system_name': 'MatCalc 钢铁数据库 v2.062',
        'database_kind': 'matcalc_original',
        'pycalphad_loadable': False,
        'supported_elements': {
            'FE', 'AL', 'B', 'C', 'CO', 'CR', 'CU', 'H', 'HF', 'LA',
            'MN', 'MO', 'N', 'NB', 'NI', 'O', 'P', 'PD', 'S', 'SI',
            'TA', 'TI', 'V', 'W', 'Y'
        },
        'recommended_phases': [],
        'recommended_use': '用于钢、双相不锈钢、马氏体时效钢等 MatCalc 场景；原文件不能直接被 pycalphad 读取。',
        'notes': '原始文件包含 MatCalc 特有语法和非 UTF-8 字符，需要额外清洗后才能尝试导入 pycalphad。',
    },
    'mc_ni_fixed_auto.tdb': {
        'system_name': 'Ni 基数据库的精简兼容版',
        'database_kind': 'simplified_bcc_fcc',
        'pycalphad_loadable': True,
        'supported_elements': {
            'AL', 'AU', 'CO', 'CR', 'CU', 'FE', 'GA', 'GE', 'HF', 'IR',
            'MO', 'NB', 'NI', 'OS', 'PD', 'PT', 'RE', 'RH', 'RU', 'SB',
            'SC', 'SN', 'TA', 'TC', 'TI', 'V', 'ZN', 'ZR'
        },
        'recommended_phases': ['FCC_A1', 'BCC_A2'],
        'recommended_use': '仅适合 Ni 基体系在 FCC/BCC 框架下的简化筛选，不等同于原始 MatCalc 超合金数据库。',
        'notes': '与原始 mc_ni_v2036 相比被大幅裁剪；已验证可被 pycalphad 加载，但只保留两相简化描述。',
    },
    'mc_ni_v2036.tdb': {
        'system_name': 'MatCalc Ni 基高温合金数据库 v2.036',
        'database_kind': 'matcalc_original',
        'pycalphad_loadable': False,
        'supported_elements': {
            'NI', 'AL', 'B', 'C', 'CO', 'CR', 'CU', 'FE', 'HF', 'LA',
            'MN', 'MO', 'N', 'NB', 'O', 'S', 'SI', 'TI', 'V', 'W',
            'Y', 'ZR'
        },
        'recommended_phases': [],
        'recommended_use': '用于 Ni 基高温合金/析出动力学的 MatCalc 场景；原文件不能直接被 pycalphad 读取。',
        'notes': '原始文件包含 MatCalc 特有语法和非 UTF-8 字符；仓库里的 mc_ni_fixed_auto.tdb 只是它的简化兼容副本。',
    },
}


def _default_tdb_dir() -> Path:
    return Path(__file__).resolve().parent.parent / 'tdb_files'


def normalize_elements(elements: Iterable[str]) -> Set[str]:
    return {element.upper() for element in elements}


def get_local_tdb_registry(tdb_dir: Optional[Path] = None) -> List[Dict]:
    base_dir = Path(tdb_dir) if tdb_dir is not None else _default_tdb_dir()
    entries: List[Dict] = []

    for filename, metadata in TDB_REGISTRY.items():
        path = base_dir / filename
        entry = {
            'filename': filename,
            'path': str(path),
            'exists': path.exists(),
            **metadata,
        }
        entries.append(entry)

    return entries


def find_best_local_tdb(
    required_elements: Iterable[str],
    tdb_dir: Optional[Path] = None,
    allow_simplified: bool = False,
) -> Optional[Dict]:
    wanted = normalize_elements(required_elements)
    priorities = {
        'full_thermodynamic': 0,
        'simplified_bcc_fcc': 1,
        'matcalc_original': 2,
    }

    candidates: List[Dict] = []
    for entry in get_local_tdb_registry(tdb_dir):
        if not entry['exists'] or not entry['pycalphad_loadable']:
            continue
        if wanted - set(entry['supported_elements']):
            continue
        if not allow_simplified and entry['database_kind'] != 'full_thermodynamic':
            continue

        score = priorities[entry['database_kind']]
        if wanted == set(entry['supported_elements']):
            score -= 1

        candidates.append((score, entry))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]['filename']))
    return candidates[0][1]
