# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, copy_metadata, collect_submodules

block_cipher = None

datas = collect_data_files('langchain')
datas += collect_data_files('chromadb')
datas += copy_metadata('tqdm')
datas += copy_metadata('regex')
datas += copy_metadata('requests')
datas += copy_metadata('packaging')
datas += copy_metadata('filelock')
datas += copy_metadata('numpy')
datas += copy_metadata('huggingface-hub')
datas += copy_metadata('safetensors')
datas += copy_metadata('pyyaml')
datas += copy_metadata('torch')
datas += copy_metadata('tokenizers')
datas += copy_metadata('opentelemetry-sdk')

hiddenimports = collect_submodules('chromadb')
hiddenimports += collect_submodules('chromadb.ingest.impl')
hiddenimports += collect_submodules('chromadb.segment.impl')
hiddenimports += collect_submodules('chromadb.segment.impl.manager')
hiddenimports += collect_submodules('chromadb.segment.impl.metadata')

a = Analysis(
    ['qualcoder/__main__.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

#splash = Splash('qualcoder.png',
#                binaries=a.binaries,
#                datas=a.datas,
#                text_pos=(10, 50),
#                text_size=12,
#                text_color='black')

exe = EXE(
    pyz,
#    splash,
    a.scripts,
    [],
    exclude_binaries=True,
    name='__main__',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='qualcoder.png'
)
coll = COLLECT(
    exe,
#    splash.binaries,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='__main__',
)

app = BUNDLE(coll,
             name='Qualcoder.app',
             icon='qualcoder/GUI/qualcoder.icns',
             bundle_identifier='org.ccbogel.qualcoder')
