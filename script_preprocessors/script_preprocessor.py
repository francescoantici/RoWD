def script_preprocessor(SCRIPT_PATH:str):
    out = ""
    with open(SCRIPT_PATH) as f:
        lines = f.readlines()
    for e in lines:
        if (e.startswith("#")) or (e == "\n") or (e == " "):
            continue 
        idx = -1
        if "#" in e:
            idx = e.index("#")
        out += e[:idx].replace("\n", "")
    return out
    
    