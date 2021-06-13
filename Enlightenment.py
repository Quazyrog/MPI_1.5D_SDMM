P = int(input("P := "))
c = int(input("c := "))

p = P
r = p // c
q = p // c**2
def divround(n, p):
    return n - (n % p)

rankmap = {}
for n in range(p):
    j = n // c
    l = n % c
    rankmap[(j, l)] = n

print("The madness starts here!")
procs = {}
for n in range(p):
    j = n // c
    l = n % c
    print(f"Process {n} has coords (j, l) = ({j}, {l})")
    print(f"  -> starts with A[{n}, :] and B[:, {n}]")
    print(f"  -> replication gives him A*[{j}, :] and B*[{j}, :]")
    inxt = ((j + q*l) % r, l)
    iprv = ((j + r - q*l) % r, l)
    print(f"  -> initially l-shifts by {q*l}, sending to {rankmap[inxt]}, receiving from {rankmap[iprv]}")
    prv = ((j + 1) % r, l)
    nxt = ((j + r - 1) % r, l)
    print(f"  -> next={rankmap[nxt]}, prev={rankmap[prv]}")
    print(f"  -> computes:")
    proc = None
    for k in range(l * q + j, (l + 1) * q + j):
        k %=  p//c
        print(f"     - C*[{k}, {j}] = A*[{k}, :] * B*[:, {j}]")
        if not proc:
            proc = {}
            proc["idx"] = n
            proc["send_to"] = rankmap[nxt]
            proc["recv_from"] = rankmap[prv]
            proc["row"] = k
            proc["col"] = j
    procs[n] = proc


# simulate several steps just in case
def draw_c(img):
    p = P
    for w in range(p//c):
        print("    ", end="")
        for k in range(p//c):
            if (w, k) in img:
                print("%02i" % img[(w, k)], end=" ")
            else:
                print("??", end=" ")
        print()

has = dict((p["idx"], p["row"]) for p in procs.values())
for exp in range(10):
    print(f"\nPowering {exp+1}/10:")
    img = {}
    for _ in range(q):
        nhas = {}
        for p in procs.values():
            assert p["idx"] == procs[p["send_to"]]["recv_from"]
            assert p["idx"] == procs[p["recv_from"]]["send_to"]
            nhas[p["idx"]] = has[p["recv_from"]]
            img[(has[p["idx"]], p["col"])] = p["idx"]
        has = nhas
    draw_c(img)
