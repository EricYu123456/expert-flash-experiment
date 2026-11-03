#!/usr/bin/env python3
"""Build an isolated offline mapping frontend; retain original hardware/ISA."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
UPSTREAM=ROOT.parent/"3dfpim-audit/upstream"
COMPILER=UPSTREAM/"3DFPIM-Compiler"
ADAPTER=ROOT/"compiler-adapter"
COMMIT="bd17a3a3c7a6a2ef98beb7a5379e2110f94df706"

def main():
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=UPSTREAM,text=True).strip()==COMMIT
    assert subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=UPSTREAM,text=True)==""
    for folder in ("src","include"):
        (ADAPTER/folder).mkdir(parents=True,exist_ok=True)
        for source in (COMPILER/folder).iterdir():
            if source.suffix in (".h",".cpp") or source.name.startswith("Makefile"):
                shutil.copy2(source,ADAPTER/folder/source.name)
    shutil.copy2(UPSTREAM/"LICENSE",ADAPTER/"LICENSE")
    shutil.copy2(COMPILER/"LICENSE_PUMA.txt",ADAPTER/"LICENSE_PUMA.txt")
    source=COMPILER/"src/partitioner.cpp"
    text=source.read_text().replace("#include <assert.h>","#include <assert.h>\n#include <cstdlib>")
    anchor="    insertLoadsAndStores();"
    insertion='''    // Offline software mapping adapter: no ISA or hardware changes.
    // Full fixed matrix inventory is retained, including unselected experts.
    if (const char* path = std::getenv("FPIM_TILE_MAP")) {
        assert(gp_ == CompilerOptions::GP_ROW_MAJOR);
        assert(N_CONSTANT_MVMUS_PER_CORE == 1);
        std::ifstream mapping(path);
        assert(mapping.good());
        unsigned maxTile = 0;
        for (unsigned c = 2; c < nVCores_; ++c) {
            unsigned tile;
            mapping >> tile;
            assert(mapping.good());
            vcore2vtile_[c] = tile + 2;
            maxTile = std::max(maxTile, tile);
        }
        nVTiles_ = maxTile + 3;
        assert(nVTiles_ <= N_MAX_TILE);
    }
    insertLoadsAndStores();'''
    assert text.count(anchor)==1
    text=text.replace(anchor,insertion)
    (ADAPTER/"src/partitioner.cpp").write_text(text)
    patch="".join(difflib.unified_diff(source.read_text().splitlines(True),text.splitlines(True),
                                    fromfile="upstream/src/partitioner.cpp",tofile="adapter/src/partitioner.cpp"))
    (ROOT/"offline_mapping_adapter.diff").write_text(patch)
    p=subprocess.run(["make","-j2"],cwd=ADAPTER/"src",capture_output=True,text=True)
    (ROOT/"adapter-build.log").write_text(p.stdout+p.stderr)
    assert p.returncode==0,p.stderr
    p=subprocess.run(["g++","-std=c++11","-O0","-g","-I"+str(ADAPTER/"include"),
                      "-I"+str(ADAPTER/"src"),str(ROOT/"native_graph_probe.cpp"),
                      "-L"+str(ADAPTER/"src"),"-l3dfpim","-o",str(ROOT/"native_graph_probe")],
                     capture_output=True,text=True)
    (ROOT/"probe-build.log").write_text(p.stdout+p.stderr)
    assert p.returncode==0,p.stderr
    compared={}
    for source in (COMPILER/"src").iterdir():
        if source.suffix not in (".cpp",".h"):continue
        target=ADAPTER/"src"/source.name
        if source.read_bytes()!=target.read_bytes():compared[source.name]=hashlib.sha256(target.read_bytes()).hexdigest()
    assert list(compared)==["partitioner.cpp"],compared
    manifest=dict(upstream_commit=COMMIT,modified_source_files=compared,
                  original_common_h_sha256=hashlib.sha256((COMPILER/"src/common.h").read_bytes()).hexdigest(),
                  adapter_common_h_sha256=hashlib.sha256((ADAPTER/"src/common.h").read_bytes()).hexdigest(),
                  change="Read fixed core-to-tile assignment before existing movement insertion; all original instruction generation and hardware parameters unchanged.",
                  patch_sha256=hashlib.sha256(patch.encode()).hexdigest())
    (ROOT/"adapter_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest))

if __name__=="__main__":main()
