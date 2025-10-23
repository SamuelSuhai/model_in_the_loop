RGC_GROUP_GROUP_ID_TO_CLASS_NAME = (
    {i: "OFF" for i in range(1, 10)}
    | {i: "ON-OFF" for i in range(10, 15)}
    | {i: "Fast ON" for i in range(15, 21)}
    | {i: "Slow ON" for i in range(21, 29)}
    | {i: "Uncertain RGC" for i in range(29, 33)}
    | {i: "AC" for i in range(33, 47)}
)