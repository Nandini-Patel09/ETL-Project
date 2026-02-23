s="abccbe"
n=len(s)//2
m=len(s)
pal = True
for i in range(n):
    l=s[i]
    r=s[m-i-1]
    print(l,r)
    if l !=r :
        pal=False
        break
if pal:
    print("yes")
else:
    print("no")
