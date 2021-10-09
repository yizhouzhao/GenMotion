# C4D Python Command Notebook


## Socket server/client
[Socket](https://www.core4d.com/ipb/forums/topic/109960-cinema4d-udp-output-limits/)

# Select object by name
```
def SelectObjectByName(obj_name):
    obj = doc.SearchObject(obj_name)
    doc.SetSelection(obj, c4d.SELECTION_NEW)
```

# Select relative/absolute position, rotation, scale

```
obj.SetRelPos(c4d.Vector(0,100,0))
```

# Get keyframe time 
```
doc.GetTime().Get()
```

# Set keyframe time
```
doc.SetTime(c4d.BaseTime(50,30))
```

# Animation reference
# https://forums.cgsociety.org/t/c4d-animation-via-python/1546556/3
# https://plugincafe.maxon.net/topic/11698/beginner-how-can-i-set-a-key-frame-and-value-to-a-cube-by-python/2