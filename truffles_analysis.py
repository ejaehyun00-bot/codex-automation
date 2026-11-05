"""Reproducible Chapter 11 truffles analysis; standard library only.
Run: python truffles_analysis.py
2SLS standard errors use the conventional homoskedastic IV covariance.
"""
from pathlib import Path
import csv, math

DATA=Path(__file__).parent/'data'/'truffles.csv'
with DATA.open(newline='') as f:
    rows=list(csv.DictReader(f))
cols={k:[float(r[k]) for r in rows] for k in rows[0]}
n=len(rows)

def transpose(A): return [list(x) for x in zip(*A)]
def mm(A,B): return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
def mv(A,v): return [sum(a*b for a,b in zip(row,v)) for row in A]
def inv(A):
    n0=len(A); M=[list(map(float,A[i]))+[float(i==j) for j in range(n0)] for i in range(n0)]
    for j in range(n0):
        pivot=max(range(j,n0),key=lambda i:abs(M[i][j]))
        if abs(M[pivot][j])<1e-14: raise ArithmeticError('singular matrix')
        M[j],M[pivot]=M[pivot],M[j]; z=M[j][j]; M[j]=[v/z for v in M[j]]
        for i in range(n0):
            if i!=j:
                z=M[i][j]; M[i]=[a-z*b for a,b in zip(M[i],M[j])]
    return [r[n0:] for r in M]
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cdf_t(t,df):
    # Student-t CDF via regularized incomplete beta (continued fraction).
    def betacf(a,b,x):
        qab=a+b; qap=a+1; qam=a-1; c=1.; d=1-qab*x/qap
        if abs(d)<1e-30:d=1e-30
        d=1/d; h=d
        for m in range(1,201):
            m2=2*m; aa=m*(b-m)*x/((qam+m2)*(a+m2)); d=1+aa*d
            if abs(d)<1e-30:d=1e-30
            c=1+aa/c
            if abs(c)<1e-30:c=1e-30
            d=1/d; h*=d*c; aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1+aa*d
            if abs(d)<1e-30:d=1e-30
            c=1+aa/c
            if abs(c)<1e-30:c=1e-30
            d=1/d; delta=d*c; h*=delta
            if abs(delta-1)<3e-14:break
        return h
    def ibeta(a,b,x):
        if x<=0:return 0.
        if x>=1:return 1.
        bt=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
        if x<(a+1)/(a+b+2): return bt*betacf(a,b,x)/a
        return 1-bt*betacf(b,a,1-x)/b
    x=df/(df+t*t); c=1-0.5*ibeta(df/2,.5,x)
    return c if t>=0 else 1-c
def p_two(t,df): return 2*(1-cdf_t(abs(t),df))
def regress(y,X,df=None):
    xt=transpose(X); xx=mm(xt,X); xi=inv(xx); b=mv(mm(xi,xt),y)
    e=[a-bb for a,bb in zip(y,mv(X,b))]; df=df or len(y)-len(b); s2=dot(e,e)/df
    se=[math.sqrt(max(0,s2*xi[i][i])) for i in range(len(b))]
    t=[b[i]/se[i] for i in range(len(b))]
    mean=sum(y)/len(y); r2=1-dot(e,e)/sum((v-mean)**2 for v in y)
    return dict(b=b,se=se,t=t,p=[p_two(v,df) for v in t],e=e,df=df,r2=r2,s2=s2)
def iv(y,X,W):
    wt=transpose(W); H=mm(mm(W,inv(mm(wt,W))),wt)
    xh=mm(H,X); xt=transpose(X); A=mm(xt,xh); Ai=inv(A); b=mv(mm(Ai,xt),mv(H,y))
    e=[a-bb for a,bb in zip(y,mv(X,b))]; df=len(y)-len(b); s2=dot(e,e)/df
    se=[math.sqrt(max(0,s2*Ai[i][i])) for i in range(len(b))]; t=[b[i]/se[i] for i in range(len(b))]
    mean=sum(y)/len(y); r2=1-dot(e,e)/sum((v-mean)**2 for v in y)
    return dict(b=b,se=se,t=t,p=[p_two(v,df) for v in t],e=e,df=df,r2=r2,s2=s2)
def show(title,m,labels):
    print(f'\n{title} (n={n}, df={m["df"]}, R2={m["r2"]:.4f})')
    print(f'{"term":<12}{"estimate":>12}{"std.error":>12}{"t":>10}{"p-value":>12}')
    for i,label in enumerate(labels):
        pv=m["p"][i]; ps=f'{pv:.4g}' if pv>0 else '<1e-300'
        print(f'{label:<12}{m["b"][i]:>12.4f}{m["se"][i]:>12.4f}{m["t"][i]:>10.3f}{ps:>12}')

one=[1.]*n; q=cols['Q']; p=cols['P']; ps=cols['PS']; di=cols['DI']; pf=cols['PF']
Z=transpose([one,ps,di,pf]); Xd=transpose([one,q,ps,di]); Xs=transpose([one,q,pf])
md=iv(p,Xd,Z); ms=iv(p,Xs,Z); od=regress(p,Xd); os=regress(p,Xs)
show('Demand 2SLS',md,['Intercept','Q','PS','DI']); show('Supply 2SLS',ms,['Intercept','Q','PF'])
show('Demand OLS',od,['Intercept','Q','PS','DI']); show('Supply OLS',os,['Intercept','Q','PF'])
means={k:sum(v)/n for k,v in cols.items()}
alpha1,alpha2,alpha3,alpha4=md['b']; beta1,beta2,beta3=ms['b']
print('\nSample means:',', '.join(f'{k}={v:.4f}' for k,v in means.items()))
print(f'Demand price elasticity at means = (1/alpha2)*(Pbar/Qbar) = {(1/alpha2)*(means["P"]/means["Q"]):.6f}')
DI,PF,PS=3.5,23.,22.
Di=alpha1+alpha3*PS+alpha4*DI; Si=beta1+beta3*PF
Qeq=(Si-Di)/(alpha2-beta2); Peq=Di+alpha2*Qeq
print(f'\nAt PS={PS:g}, DI={DI:g}, PF={PF:g}:')
print(f'Demand P = {Di:.6f} + ({alpha2:.6f}) Q')
print(f'Supply P = {Si:.6f} + ({beta2:.6f}) Q')
print(f'Structural curves equilibrium: Q={Qeq:.6f}, P={Peq:.6f}')
rfq=regress(q,Z); rfp=regress(p,Z)
show('Reduced form Q on 1,PS,DI,PF',rfq,['Intercept','PS','DI','PF']); show('Reduced form P on 1,PS,DI,PF',rfp,['Intercept','PS','DI','PF'])
z=[1.,PS,DI,PF]; Qrf=dot(z,rfq['b']); Prf=dot(z,rfp['b'])
print(f'Reduced-form predictions: Q={Qrf:.6f}, P={Prf:.6f}')
print(f'Differences structural minus reduced form: dQ={Qeq-Qrf:.6f}, dP={Peq-Prf:.6f}')
