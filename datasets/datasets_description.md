# Datasets for PiNN training

There are four datasets: diffusion equation inside the square, diffusion equation in the L-shaped domain, convection-diffusion equation, Maxwell's equation.

Scripts used for dataset generation:
1. [Diffusion equation](https://github.com/VLSF/UQNO/blob/main/datasets/PiNN_datasets/Diffusion.py) 
2. [Diffusion equation in the L-shaped domain](https://github.com/VLSF/UQNO/blob/main/datasets/PiNN_datasets/L_shaped.py)
3. [Convection-diffusion equation](https://github.com/VLSF/UQNO/blob/main/datasets/PiNN_datasets/Convection_Diffusion.py)
4. [Maxwell's equation](https://github.com/VLSF/UQNO/blob/main/datasets/PiNN_datasets/Maxwell.py)
5. [Anisotropic diffusion equation](https://github.com/VLSF/UQNO/blob/main/datasets/PiNN_datasets/Anisotropic_diffusion.py)
6. [Magnetostatics](https://github.com/VLSF/UQNO/blob/main/datasets/PiNN_datasets/magnetostatics.py)
7. [Mixed diffusion equation](https://github.com/VLSF/UQNO/blob/main/datasets/PiNN_datasets/Mixed_diffusion.py)

Datasets are available for download:
1. [Diffusion equation](https://disk.yandex.ru/d/ofuDDtCXYDiDpg)
2. [Diffusion equation in the L-shaped domain](https://disk.yandex.ru/d/2fnSN1M-CanPPw)
3. [Convection-diffusion equation](https://disk.yandex.ru/d/ZMdRFig3KaezeQ)
4. [Maxwell's equation](https://disk.yandex.ru/d/VsS0MrxlSPvl4g)
5. Anisotropic diffusion equation: [eps = 5](https://disk.yandex.ru/d/NFWX3gKxD1rtQQ), [eps = 10](https://disk.yandex.ru/d/YMh9PwMqT-r57Q), [eps = 15](https://disk.yandex.ru/d/s127Q9V_P0c4Mg), [eps = 20](https://disk.yandex.ru/d/Ye8aXLLiF2ZsfA)
6. [Magnetostatics](https://disk.yandex.ru/d/SPjPMYnB2nL1GQ)
7. Diffusion equation with mixed derivative: [eps = 0.5](https://disk.yandex.ru/d/TPwl7iyr8NqbUA), [eps = 0.7](https://disk.yandex.ru/d/gTFfZmIa9bfK-Q), [eps = 0.9](https://disk.yandex.ru/d/qus8rqEfNSQlHA), [eps = 0.99](https://disk.yandex.ru/d/tiIobWRnNsKn3w)

One can download from GoogleColab / Jupyter Notebook using the following script

```python
import requests
from urllib.parse import urlencode

base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
public_key = 'https://disk.yandex.ru/d/ofuDDtCXYDiDpg' # public link

final_url = base_url + urlencode(dict(public_key=public_key))
response = requests.get(final_url)
download_url = response.json()['href']

download_response = requests.get(download_url)
with open('Diffusion.npz', 'wb') as f:
    f.write(download_response.content)
```

Variable ```public_key``` above contains a ling to the Diffusion equation dataset. All datasets are ```.npz``` archives.

Mathematical details are given below.

## Diffusion equation

Here we convider stationary diffusion equation in the conservative form
``` math
\begin{equation}
  \begin{split}
    -&\frac{\partial}{\partial x}\left(\sigma(x, y) \frac{\partial u(x, y)}{\partial x}\right) -\frac{\partial}{\partial y}\left(\sigma(x, y) \frac{\partial u(x, y)}{\partial y}\right)  = f(x, y),\\
    &u(x, 0) = u(x, 1) = u(0, y) = u(1, y) = 0,\\
    &x, y\in(0, 1)\times(0, 1),
  \end{split}
\end{equation}
```
where
``` math
\begin{equation}
  \begin{split}
      &\sigma = \frac{z - \min z}{\max z - \min z}, z\sim \mathcal{N}\left(0, \left(I - c\Delta\right)^{-k}\right);\\
      &u = v\sin(\pi x)\sin(\pi y), v \sim \mathcal{N}\left(0, \left(I - c\Delta\right)^{-k}\right),
  \end{split}
\end{equation}
```
and the source term $f(x, y)$ is generated from diffusion coefficient $\sigma(x, y)$ and the exact solution $u(x, y)$.

**Error majorant:**
``` math
\begin{equation}
    \begin{split}
        &E[u -v] = \sqrt{\int dx dy\, \sigma(x, y) \text{grad}\,(u(x, y) - v(x, y))\cdot \text{grad}\,(u(x, y) - v(x, y))}, \\
        &E[u -v] \leq \sqrt{C_F^2(1+\beta) \int dx dy\,\left(f(x, y) + \text{div}\,w(x, y)\right)^2 + \frac{1 + \beta}{\beta \sigma(x, y)}\int dx dy\,\left(\sigma(x, y) \text{grad}\,v(x, y) - w(x, y)\right)\cdot \left(\sigma(x, y) \text{grad}\,v(x, y) - w(x, y)\right)} \\
        &C_F = 1 \big/\left(\inf_{x, y} 2\pi\sqrt{\sigma(x, y)}\right).
    \end{split}
\end{equation}
```

## Diffusion equation in the L-shaped domain

We consider, again, a stationary diffusion equation
``` math
\begin{equation}
    \begin{split}
    -&\text{div}\,\text{grad}\,u(x, y) + \left(b(x, y)\right)^2 u(x, y) = f(x, y),\\
    &\left.u(x, y)\right|_{(x, y) \in \partial \Gamma} = 0,\\
    &x, y \in \Gamma = [0, 1]^2 \setminus [0.5, 1]^2,
    \end{split}
\end{equation}
```
where 
``` math
\begin{equation}
  \begin{split}
      &b, f \sim \mathcal{N}\left(0, \left(I - c\Delta\right)^{-k}\right).
  \end{split}
\end{equation}
```
**Error majorant:**
``` math
\begin{equation}
    \begin{split}
        &E[u -v] = \sqrt{\int dx dy\left[\text{grad}\,(u(x, y) - v(x, y))\cdot \text{grad}\,(u(x, y) - v(x, y)) + \left(b(x, y)\right)^2(u(x, y) - v(x, y))^2\right]}, \\
        &E[u -v] \leq \sqrt{C_F^2(1+\beta) \int dx dy\,\left(f(x, y) - \left(b(x, y)\right)^2 v(x, y) + \text{div}\,w(x, y)\right)^2 + \frac{1 + \beta}{\beta}\int dx dy\,\left(\text{grad}\,v(x, y) - w(x, y)\right)\cdot \left(\sigma(x, y) \text{grad}\,v(x, y) - w(x, y)\right)} \\
        &C_F = 1 \big/2\pi.
    \end{split}
\end{equation}
```

## Convection-diffusion equation

For $D_x = 1$ convection-diffusion equation reads
``` math
\begin{equation}
    \begin{split}
        &\frac{\partial u(x, t)}{\partial t} - \frac{\partial^2 u(x, t)}{\partial x^2} + a\frac{\partial u(x, t)}{\partial x} = f(x), \\
        &u(x, 0) = \phi(x),\\
        &u(0, t) = u(1, t) = 0.
    \end{split}
\end{equation}
```
To solve it we consider the eigenvalue problem
``` math
\begin{equation}
    \begin{split}
        &- \frac{d^2 \psi_{\lambda}(x)}{d x^2} + a\frac{d \psi_{\lambda}(x)}{d x} = \lambda^2 \psi_{\lambda}(x), \\
        &\psi_{\lambda}(0) = \psi_{\lambda}(1) = 0,
    \end{split}
\end{equation}
```
with solution
``` math
\begin{equation}
    \psi_{k}(x) = e^{\frac{ax}{2}} \sin(\pi k x),\,\lambda_{k}^2 = \left(\pi k \right)^2 + \frac{a^2}{4}.
\end{equation}
```
Now we suppose that all PDE data is in the convenient format
``` math
\begin{equation}
    \begin{split}
        &f(x) = \sum_{k=0}^{\infty} f_k \psi_{k}(x),\\
        &\phi(x) = \sum_{k=0}^{\infty} \phi_k \psi_{k}(x),
    \end{split}
\end{equation}
```
and use standard separation of variables ansatz
``` math
\begin{equation}
    u(x, t) = \sum_{k=0}^{\infty} u_{k}(t) \psi_k(x).
\end{equation}
```

Ansatz leads to simle ODEs for each $c_{k}(t)$
``` math
\begin{equation}
    \begin{split}
      &\dot{u}_{k}(t) + \lambda_{k}^2 u_{k}(t) = f_k,\\
      &u_{k}(0) = \phi_{k}
    \end{split}
\end{equation}
```
with solution
``` math
\begin{equation}
    u_{k}(t) = \phi_k e^{-\lambda_{k}^2t} + \frac{f_k t}{1 + \lambda_{k}^2 t},
\end{equation}
```
so one can see that initial conditions decays and $u(x, t)$ approaches solution of the stationary problem.

To generate a family of PDEs we are going to draw $f_{k}$ and $\psi_k$ from normal distributions $N\left(0, \left(1 + (\pi k/ L)^s\right)^{-p}\right)$ for some positive $L, s, p$.

**Error majorant:**

For the case $a = \text{const}$ the a poteriori error estimate can be written
``` math
\begin{equation}
    \begin{split}
        &E[u - v] = \sqrt{\int dx dt\,\text{grad}\, e(x, t) \cdot \text{grad}\, e(x, t) + \frac{1}{2} \int dx\,\left(e(x, T)\right)^2},\\
        &E[u - v] \leq \sqrt{\int dx dt \,\left(y(x, t) - \text{grad}\,v(x, t)\right)\cdot \left(y(x, t) - \text{grad}\,v(x, t)\right)} + C_{F}\sqrt{\int dxdt\,\left(f(x, t) - \frac{\partial v(x, t)}{\partial t} - a\cdot \text{grad}\,v(x, t) + \text{div}\,y(x, t)\right)^2},
    \end{split}
\end{equation}
```
where $e(x, t) = u(x, t) - v(x, t)$ is the error, $v(x, t)$ approximate solution and $u(x, t)$ is the exact solution and $C = \frac{1}{\pi D_x}$, where $D_{x}$ is a number of spatial dimensions, $C_{F} = \frac{1}{\pi}$.

Source: https://arxiv.org/abs/1012.5089.


## Maxwell's equation

We consider $D=2$ Maxwell's equation in the square $x, y \in (0, 1)$

``` math
\begin{equation}
  \begin{split}
    &\frac{\partial}{\partial y}\left(\mu\left(\frac{\partial E_{y}}{\partial x} - \frac{\partial E_{x}}{\partial y}\right)\right) + E_{x} = f_{x},\\
    &-\frac{\partial}{\partial x}\left(\mu\left(\frac{\partial E_{y}}{\partial x} - \frac{\partial E_{x}}{\partial y}\right)\right) + E_{y} = f_{y},\\
    &\left.E_{x}\right|_{y=0} = \left.E_{x}\right|_{y=1} = 0,\\
    &\left.E_{y}\right|_{x=0} = \left.E_{y}\right|_{x=1} = 0.
  \end{split}
\end{equation}
```

For that we generate scalar field $A$ from $N\left(0, \left(I - \Delta\right)^{-k}\right)$ with homogeneous Neumann boundary conditions and use it to form exact solution $E_{x} = \partial_y A,\,E_{y} = -\partial_{x} A$. Next we sample $\mu$ from the same normal distribution and use all these fields to find solenoidal $f_{x}$ and $f_{y}$. This is done below.

**Error majorant:**

From [Accuracy Verification Methods: Theory and Algorithms](https://www.google.ru/books/edition/Accuracy_Verification_Methods/EFvFBAAAQBAJ?hl=en&gbpv=1&dq=Mali+repin+accuracy+verification&printsec=frontcover) (page 134, Section 4.3.1 onward) we know that the energy norm

```math
\begin{equation}
    \left(E[u - v]\right) = \sqrt{\int dxdy \left(\mu(x, y) \left(\partial_x (u(x, y) - v(x, y))_{y} - \partial_y(u(x, y)-v(x, y))_{x}\right)^2 + (u_x(x, y) - v_x(x, y))^2 + (u_y(x, y) - v_y(x, y))^2\right)},
\end{equation}
```
is bounded from above with the following error majorant
```math
\begin{equation}
    E[u - v] \leq \sqrt{\left\|f(x, y) - v(x, y) - \text{curl}\,w(x, y)\right\|^2 + \left\|\frac{1}{\sqrt{\mu(x, y)}}\left(w(x, y) - \mu(x, y)\,\text{curl}\,v(x, y)\right)\right\|^2},
\end{equation}
```
where $\text{curl}\,w(x, y) = e_x\partial_y w(x, y)-e_y\partial_{x} w(x, y)$ and $\text{curl}\,v(x, y) = \partial_x v_{y}(x, y) - \partial_y v_{x}(x, y)$, so we have a single scalar field $w(x, y)$ as certificate.


## Anisotropic diffusion equation

Similar to Diffusion equation but has additional anisotropy parameter $\epsilon$:

``` math
\begin{equation}
  \begin{split}
    -&\frac{\partial}{\partial x}\left(\sigma(x, y) \frac{\partial u(x, y)}{\partial x}\right) - \epsilon^2\frac{\partial}{\partial y}\left(\sigma(x, y) \frac{\partial u(x, y)}{\partial y}\right)  = f(x, y),\\
    &u(x, 0) = u(x, 1) = u(0, y) = u(1, y) = 0,\\
    &x, y\in(0, 1)\times(0, 1),
  \end{split}
\end{equation}
```
This parameter is also used to generate exact solution
``` math
\begin{equation}
  \begin{split}
      &\sigma = \frac{z - \min z}{\max z - \min z}, z\sim \mathcal{N}\left(0, \left(I - c\Delta\right)^{-k}\right);\\
      &u = v\sin(\pi x)\sin(\pi y), v \sim \mathcal{N}\left(0, \left(I - c\left(\partial_x^2 + \epsilon^2\partial_y^2\right)\right)^{-k}\right),
  \end{split}
\end{equation}
```
and the source term $f(x, y)$ is generated from diffusion coefficient $\sigma(x, y)$ and the exact solution $u(x, y)$.

**Error majorant:**

``` math
\begin{equation}
    \begin{split}
        &E[u -v] = \sqrt{\int dx dy\, \sigma(x, y) \text{grad}\,(u(x, y) - v(x, y))\cdot \text{grad}\,(u(x, y) - v(x, y))}, \\
        &E[u -v] \leq \sqrt{C_F^2(1+\beta) \int dx dy\,\left(f(x, y) + \text{div}\,w(x, y)\right)^2 + \frac{1 + \beta}{\beta \sigma(x, y)}\int dx dy\,\left(\left(\sigma(x, y) \text{grad}\,v(x, y) - w(x, y)\right)_x^2 + \left(\epsilon^2\sigma(x, y) \text{grad}\,v(x, y) - w(x, y)\right)_x^2 \big/ \epsilon^2\right)} \\
        &C_F = 1 \big/\left(2\pi \min\left\{1, \epsilon^2\right\}\inf_{x, y} \sqrt{\sigma(x, y)}\right).
    \end{split}
\end{equation}
```

## Magnetostatics

Equation is the same as Maxwell, but without a source term:

``` math
\begin{equation}
  \begin{split}
    &\frac{\partial}{\partial y}\left(\mu\left(\frac{\partial E_{y}}{\partial x} - \frac{\partial E_{x}}{\partial y}\right)\right) = f_{x},\\
    &-\frac{\partial}{\partial x}\left(\mu\left(\frac{\partial E_{y}}{\partial x} - \frac{\partial E_{x}}{\partial y}\right)\right) = f_{y},\\
    &\left.E_{x}\right|_{y=0} = \left.E_{x}\right|_{y=1} = 0,\\
    &\left.E_{y}\right|_{x=0} = \left.E_{y}\right|_{x=1} = 0.
  \end{split}
\end{equation}
```

This equation is interesting because error majorant is quite different.

**Error majorant:**

From [Functional a posteriori estimates for Maxwell’s equation](https://link.springer.com/article/10.1007/s10958-007-0091-8) with little adjustments we can obtain upper bound

```math
\begin{equation}
    E[u - v] \leq \frac{1}{2\pi \inf \sqrt{\mu}}\left\|f(x, y) - \text{curl}\,w(x, y)\right\|_2 + \left\|\frac{1}{\sqrt{\mu(x, y)}}\left(w(x, y) - \mu(x, y)\,\text{curl}\,v(x, y)\right)\right\|_2,
\end{equation}
```
where energy norm reads
```math
\begin{equation}
    E[u - v] = \sqrt{\int dx dy \mu(x, y) \left(\partial_x (u - v)_{y} - \partial_y(u-v)_{x}\right)^2},
\end{equation}
```
and $\text{curl}\,w(x, y) = e_x\partial_y w(x, y)-e_y\partial_{x} w(x, y)$ and $\text{curl}\,v(x, y) = \partial_x v_{y}(x, y) - \partial_y v_{x}(x, y)$, so we have a single scalar field $w(x, y)$ as certificate.

## Mixed diffusion equation

For $D_x=2$ mixed diffusion equation reads:
``` math
\begin{equation}
  \begin{split}
    -&\frac{\partial}{\partial x}\left(\sigma(x, y) \frac{\partial u(x, y)}{\partial x}\right) - \frac{\partial}{\partial y}\left(\sigma(x, y) \frac{\partial u(x, y)}{\partial y}\right) - \epsilon\frac{\partial}{\partial x}\left(\sigma(x, y) \frac{\partial u(x, y)}{\partial y}\right)-\epsilon\frac{\partial}{\partial y}\left(\sigma(x, y) \frac{\partial u(x, y)}{\partial x}\right)= f(x, y),\\
    &u(x, 0) = u(x, 1) = u(0, y) = u(1, y) = 0,\\
    &x, y\in(0, 1)\times(0, 1),
  \end{split}
\end{equation}
```
This parameter is also used to generate exact solution
``` math
\begin{equation}
  \begin{split}
      &\sigma = \frac{z - \min z}{\max z - \min z}, z\sim \mathcal{N}\left(0, \left(I - c\Delta\right)^{-k}\right);\\
      &u = v\sin(\pi x)\sin(\pi y), v \sim \mathcal{N}\left(0, \left(I - c\left(\partial_x^2 + \epsilon^2\partial_y^2\right)\right)^{-k}\right),
  \end{split}
\end{equation}
```
**Error majorant:**

``` math
\begin{equation}
    \begin{split}
        &E[u -v] = \sqrt{\int dx dy\, \sigma(x, y) \left((\partial_xu(x, y) - \partial_xv(x, y))^2 + (\partial_yu(x, y) - \partial_yv(x, y))^2\right) + 2\epsilon (\partial_yu(x, y) - \partial_yv(x, y))\left((\partial_xu(x, y) - \partial_xv(x, y))\right)}, \\
        &E[u -v]^2 \leq C_F^2(1+\beta) \int dx dy\,\left(f(x, y) + \text{div}\,w(x, y)\right)^2 +\frac{1 + \beta}{\beta}\int \frac{dx dy}{\sigma(x,y)\left(1 - \epsilon^2\right)}\left(\left(\Sigma(x, y) \text{grad}\,v(x, y) - w(x, y)\right)_x^2 + \left(\Sigma(x, y) \text{grad}\,v(x, y) - w(x, y)\right)_y^2\right) \\
        &- \frac{1 + \beta}{\beta}\int \frac{dx dy}{\sigma(x,y)(1 - \epsilon^2)}\left(2\epsilon \left(\Sigma(x, y) \text{grad}\,v(x, y) - w(x, y)\right)_x\left(\Sigma(x, y) \text{grad}\,v(x, y) - w(x, y)\right)_y\right)\\
        &C_F = 1 \big/\left(2\pi \inf_{x, y} \sqrt{\lambda_{min}\{\Sigma(x, y)\}}\right) \\
&\Sigma(x, y)=\left[
 \begin{matrix}
   \sigma(x,y) & \epsilon\sigma(x,y) \\
   \epsilon\sigma(x,y) & \sigma(x,y) 
  \end{matrix}
  \right].
    \end{split}
\end{equation}
```

## Elastoplasticity

This example is from https://www.sciencedirect.com/science/article/abs/pii/S004578259601136X.

Let $K_0$, $\mu$, $k_{\star}$, $\delta$ are positibe constants, $u$ denotes deformation vector. Given that, elastoplastic deformations are described by the following PDE

```math
\begin{split}
  &\partial_1 \sigma_{11}(u) + \partial_2 \sigma_{21}(u) + f_1 = 0,\,x_1,x_2\in\Gamma;\\
  &\partial_1 \sigma_{12}(u) + \partial_2 \sigma_{22}(u) + f_2 = 0,\,x_1,x_2\in\Gamma;\\
  &\left.u_1\right|_{\partial\Gamma} = \left.u_2\right|_{\partial\Gamma} = 0,\,\Gamma = (0, 1)^2,
\end{split}
```

where $\partial\Gamma$ is a boundary of $\Gamma$ and

```math
\begin{split}
  &\sigma(u) = K_0\left(\partial_1 u_1 + \partial_2 u_2\right) I + \gamma\left(\left\|\epsilon^{D}(u)\right\|_{F}\right) \epsilon^{D}(u);\\
  &\epsilon(u) = \begin{pmatrix}\partial_1 u_1 & \frac{1}{2}\left(\partial_1 u_2 + \partial_2 u_1\right)\\ \frac{1}{2}\left(\partial_1 u_2 + \partial_2 u_1\right) & \partial_2 u_2\end{pmatrix},\,\epsilon^{D}(u) = \epsilon(u) - \frac{1}{2}I \text{tr}\epsilon(u);\\
  &\gamma(t) = \begin{cases}2\mu,\,|t| \leq t_0 =\frac{k_{*}}{2\sqrt{\mu}};\\(2\mu-\delta)\frac{t_0}{\left|t\right|} + \delta,\,|t|>t_0.\end{cases}
\end{split}
```

**Error majorant:**

Natural energy norm for this problem reads

```math
E[u - v]^2 = \int dx \left(K_0\left(\partial_1 (u-v)_1 + \partial_2 (u-v)_2\right)^2 + \frac{\delta}{2}\left(\left(\partial_1(u-v)_1-\partial_2(u-v)_2\right)^2 + \left(\partial_1(u-v)_2+\partial_2(u-v)_1\right)^2\right)\right)
```

And the upper bound is
```math
E[u - v]^2 \leq - 2C_0 \inf_{w}\left(\frac{1}{2}\left\|w\right\|_a^2 + \mathcal{L}(\Lambda \epsilon(v); w)\right),
```

where
```math
\begin{split}
&\left\|w\right\|_a^2 = \int dx \left(K_0\left(\text{tr}\epsilon(w)\right)^2 + 2 \mu \left\|\epsilon^D(w)\right\|_F^2\right),\\
&\mathcal{L}(\Lambda \epsilon(v); w) = \int dx \left(\sum_{ij}\left(\Lambda \epsilon(v)\right)_{ij}\epsilon(w)_{ij} - f\cdot w\right),\\
&\Lambda \epsilon(v) = K_0 \text{tr} \epsilon(v) I + \gamma(\left\|\epsilon(v)\right\|_{F}) \epsilon^{D}(v),\\
&C_0 = 1+\frac{2\mu_1}{\alpha_1^{*}},\,\mu_1 = 1 - \frac{\delta}{2\mu},\,\alpha_1^{*} = \inf_{t_{11}, t_{12}, t_{22}} \frac{\frac{1}{4K_0}\left(t_{11} + t_{22}\right)^2 + \frac{1}{4\mu}\left(t_{11} - t_{22}\right)^2 + \frac{1}{\mu}t_{12}^2}{t_{11}^2 + 2t_{12}^2 + t_{22}^2}.
\end{split}
```
