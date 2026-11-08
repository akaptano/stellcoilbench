# Weak Form Derivation for Linear Elasticity

This document derives the weak form of the linear elasticity equations, starting from the strong form and arriving at the bilinear form used in the FEM implementation.

---

## 1. Strong Form

Let $\Omega \subset \mathbb{R}^3$ be the domain. The equilibrium equation and constitutive law are:

$$
-\nabla \cdot \boldsymbol{\sigma} = \mathbf{f} \quad \text{in } \Omega
$$

$$
\boldsymbol{\sigma} = \lambda \, \mathrm{tr}(\boldsymbol{\varepsilon}) \, \mathbf{I} + 2\mu \, \boldsymbol{\varepsilon}
$$

$$
\boldsymbol{\varepsilon}(\mathbf{u}) = \frac{1}{2}\bigl(\nabla\mathbf{u} + (\nabla\mathbf{u})^T\bigr)
$$

with boundary conditions: $\mathbf{u} = \mathbf{0}$ on $\Gamma_D$ (Dirichlet) and $\boldsymbol{\sigma}\mathbf{n} = \mathbf{t}$ on $\Gamma_N$ (Neumann). Here $\mathbf{f}$ is the body force and $\lambda$, $\mu$ are the Lamé parameters.

---

## 2. Principle of Virtual Work

Multiply the equilibrium equation by a test function $\mathbf{v}$ (vector-valued, vanishing on $\Gamma_D$) and integrate over $\Omega$:

$$
-\int_\Omega (\nabla \cdot \boldsymbol{\sigma}) \cdot \mathbf{v} \, dV = \int_\Omega \mathbf{f} \cdot \mathbf{v} \, dV
$$

---

## 3. Integration by Parts

Apply the product rule for the divergence:

$$
\nabla \cdot (\boldsymbol{\sigma}^T \mathbf{v}) = (\nabla \cdot \boldsymbol{\sigma}) \cdot \mathbf{v} + \boldsymbol{\sigma} : \nabla\mathbf{v}
$$

Since $\boldsymbol{\sigma}$ is symmetric, $\boldsymbol{\sigma}^T = \boldsymbol{\sigma}$, so:

$$
\nabla \cdot (\boldsymbol{\sigma} \mathbf{v}) = (\nabla \cdot \boldsymbol{\sigma}) \cdot \mathbf{v} + \boldsymbol{\sigma} : \nabla\mathbf{v}
$$

Integrating and using the divergence theorem:

$$
\int_\Omega (\nabla \cdot \boldsymbol{\sigma}) \cdot \mathbf{v} \, dV
= \int_{\partial\Omega} (\boldsymbol{\sigma}\mathbf{n}) \cdot \mathbf{v} \, dS
- \int_\Omega \boldsymbol{\sigma} : \nabla\mathbf{v} \, dV
$$

Thus:

$$
\int_\Omega \boldsymbol{\sigma} : \nabla\mathbf{v} \, dV
- \int_{\partial\Omega} (\boldsymbol{\sigma}\mathbf{n}) \cdot \mathbf{v} \, dS
= \int_\Omega \mathbf{f} \cdot \mathbf{v} \, dV
$$

The boundary integral is the **traction integral**. It must be retained in the weak form. Partition $\partial\Omega = \Gamma_D \cup \Gamma_N$:

- **$\Gamma_D$ (Dirichlet)**: Where displacement is prescribed, e.g. $\mathbf{u} = \mathbf{0}$. Test functions vanish here: $\mathbf{v} = \mathbf{0}$ on $\Gamma_D$, so
  $$\int_{\Gamma_D} (\boldsymbol{\sigma}\mathbf{n}) \cdot \mathbf{v} \, dS = 0.$$

- **$\Gamma_N$ (Neumann)**: Where traction $\mathbf{t} = \boldsymbol{\sigma}\mathbf{n}$ is prescribed. On $\Gamma_N$, the traction integral is *known* (a linear functional in $\mathbf{v}$) and must appear on the **right-hand side**:
  $$\int_{\Gamma_N} (\boldsymbol{\sigma}\mathbf{n}) \cdot \mathbf{v} \, dS = \int_{\Gamma_N} \mathbf{t} \cdot \mathbf{v} \, dS.$$

Hence the weak form becomes:

$$
\int_\Omega \boldsymbol{\sigma} : \nabla\mathbf{v} \, dV
= \int_\Omega \mathbf{f} \cdot \mathbf{v} \, dV
+ \int_{\Gamma_N} \mathbf{t} \cdot \mathbf{v} \, dS
$$

For **homogeneous Neumann** (free surface): $\mathbf{t} = \mathbf{0}$ on $\Gamma_N$. Then
$\int_{\Gamma_N} \mathbf{t} \cdot \mathbf{v} \, dS = 0$, and we obtain:

$$
\int_\Omega \boldsymbol{\sigma} : \nabla\mathbf{v} \, dV = \int_\Omega \mathbf{f} \cdot \mathbf{v} \, dV
$$

The traction integral does not vanish because we "dropped" it — we impose $\mathbf{t} = \mathbf{0}$, so it evaluates to zero. The FEM RHS needs no boundary term when $\mathbf{t} = \mathbf{0}$. When $\mathbf{t} \neq \mathbf{0}$, we must add $\int_{\Gamma_N} \mathbf{t} \cdot \mathbf{v} \, dS$ to the load vector.

---

## 4. Symmetry: $\boldsymbol{\sigma} : \nabla\mathbf{v} = \boldsymbol{\sigma} : \boldsymbol{\varepsilon}(\mathbf{v})$

The stress tensor $\boldsymbol{\sigma}$ is symmetric. Decompose the gradient into symmetric and antisymmetric parts:

$$
\nabla\mathbf{v} = \boldsymbol{\varepsilon}(\mathbf{v}) + \boldsymbol{\omega}(\mathbf{v}),
\quad
\boldsymbol{\varepsilon}(\mathbf{v}) = \frac{1}{2}(\nabla\mathbf{v} + (\nabla\mathbf{v})^T),
\quad
\boldsymbol{\omega}(\mathbf{v}) = \frac{1}{2}(\nabla\mathbf{v} - (\nabla\mathbf{v})^T)
$$

The double contraction of a symmetric tensor with an antisymmetric tensor is zero:
$\boldsymbol{\sigma} : \boldsymbol{\omega} = 0$. Hence:

$$
\boldsymbol{\sigma} : \nabla\mathbf{v} = \boldsymbol{\sigma} : \boldsymbol{\varepsilon}(\mathbf{v})
$$

So we have:

$$
\int_\Omega \boldsymbol{\sigma} : \boldsymbol{\varepsilon}(\mathbf{v}) \, dV = \int_\Omega \mathbf{f} \cdot \mathbf{v} \, dV
$$

---

## 5. Substitute the Constitutive Law

$$
\boldsymbol{\sigma} = \lambda \, \mathrm{tr}(\boldsymbol{\varepsilon}(\mathbf{u})) \, \mathbf{I} + 2\mu \, \boldsymbol{\varepsilon}(\mathbf{u})
$$

Substitute into the weak form:

$$
\int_\Omega \bigl[ \lambda \, \mathrm{tr}(\boldsymbol{\varepsilon}(\mathbf{u})) \, \mathbf{I} + 2\mu \, \boldsymbol{\varepsilon}(\mathbf{u}) \bigr] : \boldsymbol{\varepsilon}(\mathbf{v}) \, dV
= \int_\Omega \mathbf{f} \cdot \mathbf{v} \, dV
$$

---

## 6. Simplify the First Term: $\mathbf{I} : \boldsymbol{\varepsilon}(\mathbf{v})$

The identity tensor satisfies $\mathbf{I} : \boldsymbol{\varepsilon} = \mathrm{tr}(\boldsymbol{\varepsilon})$. Therefore:

$$
\bigl( \lambda \, \mathrm{tr}(\boldsymbol{\varepsilon}(\mathbf{u})) \, \mathbf{I} \bigr) : \boldsymbol{\varepsilon}(\mathbf{v})
= \lambda \, \mathrm{tr}(\boldsymbol{\varepsilon}(\mathbf{u})) \, \mathbf{I} : \boldsymbol{\varepsilon}(\mathbf{v})
= \lambda \, \mathrm{tr}(\boldsymbol{\varepsilon}(\mathbf{u})) \, \mathrm{tr}(\boldsymbol{\varepsilon}(\mathbf{v}))
$$

Since $\mathrm{tr}(\boldsymbol{\varepsilon}) = \nabla \cdot \mathbf{u}$ (and similarly for $\mathbf{v}$):

$$
\lambda \, \mathrm{tr}(\boldsymbol{\varepsilon}(\mathbf{u})) \, \mathrm{tr}(\boldsymbol{\varepsilon}(\mathbf{v}))
= \lambda \, (\nabla \cdot \mathbf{u}) \, (\nabla \cdot \mathbf{v})
$$

---

## 7. Simplify the Second Term

$$
\bigl( 2\mu \, \boldsymbol{\varepsilon}(\mathbf{u}) \bigr) : \boldsymbol{\varepsilon}(\mathbf{v})
= 2\mu \, \boldsymbol{\varepsilon}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{v})
$$

---

## 8. Final Weak Form

Combining both terms:

$$
\boxed{
\int_\Omega \Bigl( \lambda \, (\nabla \cdot \mathbf{u}) \, (\nabla \cdot \mathbf{v})
+ 2\mu \, \boldsymbol{\varepsilon}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{v}) \Bigr) \, dV
= \int_\Omega \mathbf{f} \cdot \mathbf{v} \, dV
+ \int_{\Gamma_N} \mathbf{t} \cdot \mathbf{v} \, dS
}
$$

This is the full weak form including the traction integral on the Neumann boundary. The bilinear form is

$$
a(\mathbf{u}, \mathbf{v}) = \int_\Omega \Bigl( \lambda \, (\nabla \cdot \mathbf{u}) \, (\nabla \cdot \mathbf{v})
+ 2\mu \, \boldsymbol{\varepsilon}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{v}) \Bigr) \, dV
$$

and the linear form (load) is

$$
L(\mathbf{v}) = \int_\Omega \mathbf{f} \cdot \mathbf{v} \, dV
+ \int_{\Gamma_N} \mathbf{t} \cdot \mathbf{v} \, dS.
$$

**FEM setup**: Solve $a(\mathbf{u}, \mathbf{v}) = L(\mathbf{v})$ for all $\mathbf{v}$ in the test space (vanishing on $\Gamma_D$), with $\mathbf{u}$ satisfying the Dirichlet conditions.

---

## 9. Relation to the StellCoilBench Implementation

The StellCoilBench structural solver uses:

- **$\Gamma_D$**: Boundary facets (or nodes) in the bottom 15% of the z-range — clamped, $\mathbf{u} = \mathbf{0}$.
- **$\Gamma_N$**: Remainder of the boundary — free surface with **homogeneous Neumann** $\mathbf{t} = \mathbf{0}$.

Because $\mathbf{t} = \mathbf{0}$ on $\Gamma_N$, the traction integral $\int_{\Gamma_N} \mathbf{t} \cdot \mathbf{v} \, dS$ is zero. The code therefore sets

$$
L(\mathbf{v}) = \int_\Omega \mathbf{f} \cdot \mathbf{v} \, dV
$$

with no boundary term. The formulation is correct. If nonzero traction (e.g. pressure, contact forces) were ever prescribed on part of the boundary, the implementation would need to add $\int_{\Gamma_N} \mathbf{t} \cdot \mathbf{v} \, dS$ to the load vector.
