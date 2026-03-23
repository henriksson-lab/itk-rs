//! Optimization algorithms.
//!
//! | Optimizer | ITK analog |
//! |---|---|
//! | [`GradientDescentOptimizer`] | `GradientDescentOptimizerv4` |
//! | [`RegularStepGradientDescentOptimizer`] | `RegularStepGradientDescentOptimizerv4` |
//! | [`LBFGSOptimizer`] | `LBFGSOptimizerv4` |
//! | [`AmoebaOptimizer`] | `AmoebaOptimizer` (Nelder-Mead) |
//! | [`PowellOptimizer`] | `PowellOptimizer` |
//! | [`ExhaustiveOptimizer`] | `ExhaustiveOptimizer` |
//! | [`ConjugateGradientOptimizer`] | `ConjugateGradientLineSearchOptimizerv4` |

// ---------------------------------------------------------------------------
// Optimizer trait
// ---------------------------------------------------------------------------

/// Common interface for function minimizers.
pub trait Optimizer {
    /// Run the optimization. Returns final parameter vector.
    fn optimize<F>(&self, initial: &[f64], cost: F) -> Vec<f64>
    where F: Fn(&[f64]) -> f64;
}

// ---------------------------------------------------------------------------
// GradientDescentOptimizer
// ---------------------------------------------------------------------------

/// Steepest-descent optimizer.
/// Analog to `itk::GradientDescentOptimizerv4`.
pub struct GradientDescentOptimizer {
    pub learning_rate: f64,
    pub max_iterations: usize,
    pub convergence_threshold: f64,
}

impl GradientDescentOptimizer {
    pub fn new(learning_rate: f64, max_iterations: usize) -> Self {
        Self { learning_rate, max_iterations, convergence_threshold: 1e-6 }
    }
}

fn numerical_gradient(cost: &impl Fn(&[f64]) -> f64, x: &[f64], h: f64) -> Vec<f64> {
    let mut grad = vec![0.0f64; x.len()];
    let f0 = cost(x);
    let mut xp = x.to_vec();
    for i in 0..x.len() {
        xp[i] += h;
        let fp = cost(&xp);
        grad[i] = (fp - f0) / h;
        xp[i] = x[i];
    }
    grad
}

impl Optimizer for GradientDescentOptimizer {
    fn optimize<F>(&self, initial: &[f64], cost: F) -> Vec<f64>
    where F: Fn(&[f64]) -> f64 {
        let mut x = initial.to_vec();
        let h = 1e-5;
        for _ in 0..self.max_iterations {
            let grad = numerical_gradient(&cost, &x, h);
            let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if grad_norm < self.convergence_threshold { break; }
            for i in 0..x.len() {
                x[i] -= self.learning_rate * grad[i];
            }
        }
        x
    }
}

// ---------------------------------------------------------------------------
// RegularStepGradientDescentOptimizer
// ---------------------------------------------------------------------------

/// Gradient descent with step-halving on non-decrease.
/// Analog to `itk::RegularStepGradientDescentOptimizerv4`.
pub struct RegularStepGradientDescentOptimizer {
    pub learning_rate: f64,
    pub min_step_length: f64,
    pub max_iterations: usize,
    pub relaxation_factor: f64,
}

impl RegularStepGradientDescentOptimizer {
    pub fn new(learning_rate: f64, max_iterations: usize) -> Self {
        Self { learning_rate, min_step_length: 1e-6, max_iterations, relaxation_factor: 0.5 }
    }
}

impl Optimizer for RegularStepGradientDescentOptimizer {
    fn optimize<F>(&self, initial: &[f64], cost: F) -> Vec<f64>
    where F: Fn(&[f64]) -> f64 {
        let mut x = initial.to_vec();
        let mut step = self.learning_rate;
        let h = 1e-5;
        for _ in 0..self.max_iterations {
            if step < self.min_step_length { break; }
            let f0 = cost(&x);
            let grad = numerical_gradient(&cost, &x, h);
            let mut x_new = x.clone();
            for i in 0..x.len() { x_new[i] -= step * grad[i]; }
            let f_new = cost(&x_new);
            if f_new < f0 {
                x = x_new;
            } else {
                step *= self.relaxation_factor;
            }
        }
        x
    }
}

// ---------------------------------------------------------------------------
// LBFGSOptimizer
// ---------------------------------------------------------------------------

/// Limited-memory BFGS optimizer.
/// Analog to `itk::LBFGSOptimizerv4`.
pub struct LBFGSOptimizer {
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub memory_size: usize,
}

impl LBFGSOptimizer {
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations, convergence_threshold: 1e-6, memory_size: 10 }
    }
}

impl Optimizer for LBFGSOptimizer {
    fn optimize<F>(&self, initial: &[f64], cost: F) -> Vec<f64>
    where F: Fn(&[f64]) -> f64 {
        // L-BFGS two-loop recursion
        let n = initial.len();
        let h = 1e-5;
        let mut x = initial.to_vec();
        let mut ss: Vec<Vec<f64>> = Vec::new(); // parameter differences
        let mut ys: Vec<Vec<f64>> = Vec::new(); // gradient differences
        let mut g_prev = numerical_gradient(&cost, &x, h);

        for iter in 0..self.max_iterations {
            let g = numerical_gradient(&cost, &x, h);
            let g_norm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
            if g_norm < self.convergence_threshold { break; }

            // Two-loop L-BFGS
            let m = ss.len();
            let mut q = g.clone();
            let mut alphas = vec![0.0f64; m];
            for i in (0..m).rev() {
                let rho_i: f64 = 1.0 / ys[i].iter().zip(ss[i].iter()).map(|(y, s)| y * s).sum::<f64>().max(1e-12);
                alphas[i] = rho_i * ss[i].iter().zip(q.iter()).map(|(s, qi)| s * qi).sum::<f64>();
                for j in 0..n { q[j] -= alphas[i] * ys[i][j]; }
            }
            // Scale by Hessian approx
            let mut r = q.clone();
            if m > 0 {
                let sy = ys[m-1].iter().zip(ss[m-1].iter()).map(|(y, s)| y * s).sum::<f64>();
                let yy = ys[m-1].iter().map(|y| y * y).sum::<f64>();
                let scale = sy / yy.max(1e-12);
                for ri in &mut r { *ri *= scale; }
            }
            for i in 0..m {
                let rho_i: f64 = 1.0 / ys[i].iter().zip(ss[i].iter()).map(|(y, s)| y * s).sum::<f64>().max(1e-12);
                let beta = rho_i * ys[i].iter().zip(r.iter()).map(|(yi, ri)| yi * ri).sum::<f64>();
                for j in 0..n { r[j] += ss[i][j] * (alphas[i] - beta); }
            }

            // Line search (backtracking Armijo)
            let f0 = cost(&x);
            let mut step = 1.0f64;
            let c = 1e-4;
            let gd: f64 = g.iter().zip(r.iter()).map(|(gi, ri)| gi * ri).sum::<f64>();
            let mut x_new = x.clone();
            for _ in 0..20 {
                for j in 0..n { x_new[j] = x[j] - step * r[j]; }
                if cost(&x_new) <= f0 - c * step * gd.abs() { break; }
                step *= 0.5;
            }

            let s: Vec<f64> = (0..n).map(|j| x_new[j] - x[j]).collect();
            let g_new = numerical_gradient(&cost, &x_new, h);
            let y: Vec<f64> = (0..n).map(|j| g_new[j] - g[j]).collect();

            x = x_new;
            ss.push(s);
            ys.push(y);
            if ss.len() > self.memory_size { ss.remove(0); ys.remove(0); }

            if iter == 0 { let _ = g_prev; }
            g_prev = g_new;
        }
        x
    }
}

// ---------------------------------------------------------------------------
// AmoebaOptimizer (Nelder-Mead)
// ---------------------------------------------------------------------------

/// Nelder-Mead simplex optimizer.
/// Analog to `itk::AmoebaOptimizer`.
pub struct AmoebaOptimizer {
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub initial_simplex_delta: f64,
}

impl AmoebaOptimizer {
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations, convergence_threshold: 1e-6, initial_simplex_delta: 0.05 }
    }
}

impl Optimizer for AmoebaOptimizer {
    fn optimize<F>(&self, initial: &[f64], cost: F) -> Vec<f64>
    where F: Fn(&[f64]) -> f64 {
        let n = initial.len();
        // Build initial simplex
        let mut simplex: Vec<Vec<f64>> = vec![initial.to_vec()];
        for i in 0..n {
            let mut s = initial.to_vec();
            s[i] += if s[i] == 0.0 { 0.00025 } else { self.initial_simplex_delta * s[i].abs() };
            simplex.push(s);
        }
        let mut values: Vec<f64> = simplex.iter().map(|s| cost(s)).collect();

        for _ in 0..self.max_iterations {
            // Sort by value
            let mut order: Vec<usize> = (0..=n).collect();
            order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

            let best = order[0];
            let worst = order[n];
            let second_worst = order[n - 1];

            // Convergence check
            let range: f64 = values[worst] - values[best];
            if range < self.convergence_threshold { break; }

            // Centroid of all but worst
            let mut centroid = vec![0.0f64; n];
            for &i in &order[..n] {
                for j in 0..n { centroid[j] += simplex[i][j]; }
            }
            for j in 0..n { centroid[j] /= n as f64; }

            // Reflection
            let reflected: Vec<f64> = (0..n).map(|j| centroid[j] + (centroid[j] - simplex[worst][j])).collect();
            let f_ref = cost(&reflected);

            if f_ref < values[best] {
                // Expansion
                let expanded: Vec<f64> = (0..n).map(|j| centroid[j] + 2.0 * (reflected[j] - centroid[j])).collect();
                let f_exp = cost(&expanded);
                if f_exp < f_ref {
                    simplex[worst] = expanded;
                    values[worst] = f_exp;
                } else {
                    simplex[worst] = reflected;
                    values[worst] = f_ref;
                }
            } else if f_ref < values[second_worst] {
                simplex[worst] = reflected;
                values[worst] = f_ref;
            } else {
                // Contraction
                let contracted: Vec<f64> = (0..n).map(|j| centroid[j] + 0.5 * (simplex[worst][j] - centroid[j])).collect();
                let f_con = cost(&contracted);
                if f_con < values[worst] {
                    simplex[worst] = contracted;
                    values[worst] = f_con;
                } else {
                    // Shrink
                    for i in 1..=n {
                        let idx = order[i];
                        simplex[idx] = (0..n).map(|j| simplex[best][j] + 0.5 * (simplex[idx][j] - simplex[best][j])).collect();
                        values[idx] = cost(&simplex[idx]);
                    }
                }
            }
        }

        let best = (0..=n).min_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap()).unwrap();
        simplex[best].clone()
    }
}

// ---------------------------------------------------------------------------
// PowellOptimizer
// ---------------------------------------------------------------------------

/// Powell's direction-set method.
/// Analog to `itk::PowellOptimizer`.
pub struct PowellOptimizer {
    pub max_iterations: usize,
    pub max_line_iterations: usize,
    pub step_tolerance: f64,
    pub value_tolerance: f64,
}

impl PowellOptimizer {
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations, max_line_iterations: 100, step_tolerance: 1e-6, value_tolerance: 1e-6 }
    }
}

impl Optimizer for PowellOptimizer {
    fn optimize<F>(&self, initial: &[f64], cost: F) -> Vec<f64>
    where F: Fn(&[f64]) -> f64 {
        let n = initial.len();
        let mut x = initial.to_vec();
        let mut dirs: Vec<Vec<f64>> = (0..n).map(|i| {
            let mut d = vec![0.0f64; n]; d[i] = 1.0; d
        }).collect();

        for _ in 0..self.max_iterations {
            let x_start = x.clone();
            for i in 0..n {
                // Line minimization along dirs[i]
                let d = dirs[i].clone();
                let line_cost = |t: f64| -> f64 {
                    let xp: Vec<f64> = x.iter().zip(d.iter()).map(|(&xi, &di)| xi + t * di).collect();
                    cost(&xp)
                };
                let t = golden_section(&line_cost, -1.0, 1.0, self.max_line_iterations);
                for j in 0..n { x[j] += t * d[j]; }
            }

            // Add new direction (x_end - x_start)
            let new_dir: Vec<f64> = (0..n).map(|j| x[j] - x_start[j]).collect();
            let norm: f64 = new_dir.iter().map(|d| d * d).sum::<f64>().sqrt();
            if norm < self.step_tolerance { break; }
            let new_dir_norm: Vec<f64> = new_dir.iter().map(|d| d / norm).collect();
            dirs.remove(0);
            dirs.push(new_dir_norm);
        }
        x
    }
}

fn golden_section(f: &impl Fn(f64) -> f64, a: f64, b: f64, max_iter: usize) -> f64 {
    let phi = (5.0f64.sqrt() - 1.0) / 2.0;
    let (mut lo, mut hi) = (a, b);
    let mut x1 = hi - phi * (hi - lo);
    let mut x2 = lo + phi * (hi - lo);
    let mut f1 = f(x1); let mut f2 = f(x2);
    for _ in 0..max_iter {
        if (hi - lo).abs() < 1e-8 { break; }
        if f1 < f2 {
            hi = x2; x2 = x1; f2 = f1;
            x1 = hi - phi * (hi - lo); f1 = f(x1);
        } else {
            lo = x1; x1 = x2; f1 = f2;
            x2 = lo + phi * (hi - lo); f2 = f(x2);
        }
    }
    (lo + hi) / 2.0
}

// ---------------------------------------------------------------------------
// ExhaustiveOptimizer
// ---------------------------------------------------------------------------

/// Grid-search optimizer.
/// Analog to `itk::ExhaustiveOptimizer`.
pub struct ExhaustiveOptimizer {
    pub number_of_steps: Vec<usize>,
    pub step_length: f64,
}

impl ExhaustiveOptimizer {
    pub fn new(number_of_steps: Vec<usize>, step_length: f64) -> Self {
        Self { number_of_steps, step_length }
    }
}

impl Optimizer for ExhaustiveOptimizer {
    fn optimize<F>(&self, initial: &[f64], cost: F) -> Vec<f64>
    where F: Fn(&[f64]) -> f64 {
        let n = initial.len().min(self.number_of_steps.len());
        let mut best_x = initial.to_vec();
        let mut best_val = f64::MAX;

        // Recursive grid search
        fn search(
            dim: usize, n: usize, x: &mut Vec<f64>, initial: &[f64],
            steps: &[usize], step_len: f64, best_x: &mut Vec<f64>, best_val: &mut f64,
            cost: &impl Fn(&[f64]) -> f64,
        ) {
            if dim == n {
                let v = cost(x);
                if v < *best_val { *best_val = v; *best_x = x.clone(); }
                return;
            }
            let n_steps = steps[dim];
            for s in 0..=(2 * n_steps) {
                x[dim] = initial[dim] + (s as f64 - n_steps as f64) * step_len;
                search(dim + 1, n, x, initial, steps, step_len, best_x, best_val, cost);
            }
        }

        let mut x = initial.to_vec();
        search(0, n, &mut x, initial, &self.number_of_steps, self.step_length, &mut best_x, &mut best_val, &cost);
        best_x
    }
}

// ---------------------------------------------------------------------------
// ConjugateGradientOptimizer
// ---------------------------------------------------------------------------

/// Fletcher-Reeves conjugate gradient optimizer.
/// Analog to `itk::ConjugateGradientLineSearchOptimizerv4`.
pub struct ConjugateGradientOptimizer {
    pub max_iterations: usize,
    pub convergence_threshold: f64,
}

impl ConjugateGradientOptimizer {
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations, convergence_threshold: 1e-6 }
    }
}

impl Optimizer for ConjugateGradientOptimizer {
    fn optimize<F>(&self, initial: &[f64], cost: F) -> Vec<f64>
    where F: Fn(&[f64]) -> f64 {
        let n = initial.len();
        let h = 1e-5;
        let mut x = initial.to_vec();
        let mut g = numerical_gradient(&cost, &x, h);
        let mut d: Vec<f64> = g.iter().map(|gi| -gi).collect();

        for iter in 0..self.max_iterations {
            let g_norm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
            if g_norm < self.convergence_threshold { break; }

            // Line search along d
            let line_cost = |t: f64| -> f64 {
                let xp: Vec<f64> = x.iter().zip(d.iter()).map(|(&xi, &di)| xi + t * di).collect();
                cost(&xp)
            };
            let t = golden_section(&line_cost, 0.0, 1.0, 100);
            for j in 0..n { x[j] += t * d[j]; }

            let g_new = numerical_gradient(&cost, &x, h);
            // Fletcher-Reeves beta
            let beta = if iter == 0 { 0.0 } else {
                let num: f64 = g_new.iter().map(|gi| gi * gi).sum();
                let den: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().max(1e-12);
                num / den
            };
            d = (0..n).map(|j| -g_new[j] + beta * d[j]).collect();
            g = g_new;
        }
        x
    }
}

// ---------------------------------------------------------------------------
// LBFGSB (bounded L-BFGS)
// ---------------------------------------------------------------------------

/// Bounded L-BFGS optimizer.
/// Analog to `itk::LBFGSBOptimizerv4`.
pub struct LBFGSBOptimizer {
    pub max_iterations: usize,
    pub lower_bound: Option<Vec<f64>>,
    pub upper_bound: Option<Vec<f64>>,
}

impl LBFGSBOptimizer {
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations, lower_bound: None, upper_bound: None }
    }
}

impl Optimizer for LBFGSBOptimizer {
    fn optimize<F>(&self, initial: &[f64], cost: F) -> Vec<f64>
    where F: Fn(&[f64]) -> f64 {
        // Delegate to L-BFGS, applying box projection after each step
        let lbfgs = LBFGSOptimizer::new(self.max_iterations);
        let n = initial.len();
        let lb = self.lower_bound.clone().unwrap_or_else(|| vec![f64::NEG_INFINITY; n]);
        let ub = self.upper_bound.clone().unwrap_or_else(|| vec![f64::INFINITY; n]);
        let projected_cost = |x: &[f64]| -> f64 {
            let xp: Vec<f64> = x.iter().enumerate().map(|(i, &xi)| xi.clamp(lb[i], ub[i])).collect();
            cost(&xp)
        };
        let result = lbfgs.optimize(initial, projected_cost);
        result.into_iter().enumerate().map(|(i, xi)| xi.clamp(lb[i], ub[i])).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    #[test]
    fn gradient_descent_minimizes_quadratic() {
        let opt = GradientDescentOptimizer::new(0.1, 200);
        let result = opt.optimize(&[1.0, 2.0], quadratic);
        assert!(quadratic(&result) < 0.01, "not minimized: {:?}", result);
    }

    #[test]
    fn amoeba_minimizes_quadratic() {
        let opt = AmoebaOptimizer::new(500);
        let result = opt.optimize(&[1.0, 2.0], quadratic);
        assert!(quadratic(&result) < 0.01, "not minimized: {:?}", result);
    }

    #[test]
    fn exhaustive_finds_minimum() {
        // For a discrete quadratic the exhaustive should find (0,0)
        let opt = ExhaustiveOptimizer::new(vec![5, 5], 0.5);
        let result = opt.optimize(&[0.0, 0.0], quadratic);
        assert!(quadratic(&result) < 0.01, "not minimized: {:?}", result);
    }

    #[test]
    fn conjugate_gradient_minimizes_quadratic() {
        let opt = ConjugateGradientOptimizer::new(500);
        let result = opt.optimize(&[1.0, 2.0], quadratic);
        assert!(quadratic(&result) < 0.01, "not minimized: {:?}", result);
    }
}
