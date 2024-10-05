use std::fmt;
use std::str::FromStr;

// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[derive(Debug, PartialEq, Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<Vec<f64>>, // Изменено на f64 для работы с дробями
}

impl Matrix {
    fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    fn determinant(&self) -> Result<f64, String> {
        if self.rows != self.cols {
            return Err("Determinant can only be calculated for square matrices".to_string());
        }

        if self.rows == 1 {
            return Ok(self.data[0][0]);
        }

        if self.rows == 2 {
            return Ok(self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]);
        }

        let mut det = 0.0;
        for j in 0..self.cols {
            let mut submatrix = Matrix::new(self.rows - 1, self.cols - 1);
            for i in 1..self.rows {
                let mut k = 0;
                for l in 0..self.cols {
                    if l != j {
                        submatrix.data[i - 1][k] = self.data[i][l];
                        k += 1;
                    }
                }
            }
            det += self.data[0][j] * submatrix.determinant()? * if j % 2 == 0 { 1.0 } else { -1.0 };
        }

        Ok(det)
    }

    fn transpose(&self) -> Matrix {
        let mut transposed = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed.data[j][i] = self.data[i][j];
            }
        }
        transposed
    }

    fn inverse(&self) -> Result<Matrix, String> {
        let det = self.determinant()?;
        if det == 0.0 {
            return Err("Matrix is not invertible (determinant is 0)".to_string());
        }

        if self.rows == 1 {
            return Ok(Matrix {
                rows: 1,
                cols: 1,
                data: vec![vec![1.0 / det]],
            });
        }

        let mut adjugate = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let mut submatrix = Matrix::new(self.rows - 1, self.cols - 1);
                let mut row_idx = 0;
                for k in 0..self.rows {
                    if k == i {
                        continue;
                    }
                    let mut col_idx = 0;
                    for l in 0..self.cols {
                        if l == j {
                            continue;
                        }
                        submatrix.data[row_idx][col_idx] = self.data[k][l];
                        col_idx += 1;
                    }
                    row_idx += 1;
                }
                adjugate.data[i][j] = submatrix.determinant()? * if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            }
        }

        let inverse = adjugate.transpose() * (1.0 / det); // Умножение на константу
        Ok(inverse)
    }

    fn gaussian_elimination(&self) -> Result<Vec<f64>, String> {
        if self.rows + 1 != self.cols {
            return Err("Invalid matrix dimensions for Gaussian elimination".to_string());
        }

        let mut augmented_matrix = self.clone(); // Работаем с копией, чтобы не изменять исходную матрицу

        // Прямой ход (приведение к треугольному виду)
        for i in 0..self.rows {
            // Находим максимальный элемент в столбце i (начиная с i-й строки)
            let mut max_row = i;
            for k in i + 1..self.rows {
                if augmented_matrix.data[k][i].abs() > augmented_matrix.data[max_row][i].abs() {
                    max_row = k;
                }
            }

            // Меняем строки местами, если нужно
            if max_row != i {
                augmented_matrix.data.swap(i, max_row);
            }

            // Обнуляем элементы ниже i-го элемента в столбце i
            for k in i + 1..self.rows {
                let factor = augmented_matrix.data[k][i] / augmented_matrix.data[i][i];
                for j in i..self.cols {
                    augmented_matrix.data[k][j] -= factor * augmented_matrix.data[i][j];
                }
            }
        }

        // Обратный ход (нахождение решения)
        let mut solutions = vec![0.0; self.rows];
        for i in (0..self.rows).rev() {
            solutions[i] = augmented_matrix.data[i][self.cols - 1];
            for j in i + 1..self.rows {
                solutions[i] -= augmented_matrix.data[i][j] * solutions[j];
            }
            solutions[i] /= augmented_matrix.data[i][i];
        }

        Ok(solutions)
    }

    fn cramer_rule(&self) -> Result<Vec<f64>, String> {
        let n = self.rows; // Количество уравнений (и неизвестных)

        if n != self.cols - 1 {
            return Err("Invalid matrix dimensions for Cramer's rule".to_string());
        }

        let mut core_matrix = Matrix::new(n, n); // Матрица коэффициентов
        for i in 0..n {
            for j in 0..n {
                core_matrix.data[i][j] = self.data[i][j];
            }
        }

        let det_a = core_matrix.determinant()?; // Определитель основной матрицы

        if det_a == 0.0 {
            return Err("System has no unique solution (determinant is 0)".to_string());
        }

        let mut solutions = vec![0.0; n];
        for i in 0..n {
            let mut temp_matrix = core_matrix.clone(); // Копируем матрицу коэффициентов
            for j in 0..n {
                temp_matrix.data[j][i] = self.data[j][n]; // Подставляем столбец свободных членов
            }
            solutions[i] = temp_matrix.determinant()? / det_a;
        }

        Ok(solutions)
    }
}

impl FromStr for Matrix {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let rows: Vec<&str> = s.trim().split('\n').collect();
        let rows_count = rows.len();
        if rows_count == 0 {
            return Err("Empty matrix string".to_string());
        }

        let cols_count = rows[0].trim().split_whitespace().count();
        if cols_count == 0 {
            return Err("Empty matrix row".to_string());
        }

        let mut data = Vec::with_capacity(rows_count);
        for row_str in rows {
            let row: Vec<f64> = row_str // Изменено на f64
                .trim()
                .split_whitespace()
                .map(|s| s.parse().map_err(|_| "Invalid number in matrix".to_string()))
                .collect::<Result<_, _>>()?;
            if row.len() != cols_count {
                return Err("Inconsistent number of columns".to_string());
            }
            data.push(row);
        }

        Ok(Matrix {
            rows: rows_count,
            cols: cols_count,
            data,
        })
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let max_widths: Vec<usize> = (0..self.cols)
            .map(|j| {
                self.data
                    .iter()
                    .map(|row| row[j].to_string().len())
                    .max()
                    .unwrap_or(0)
            })
            .collect();

        for row in &self.data {
            for (j, &val) in row.iter().enumerate() {
                write!(f, " {:^width$} ", val, width = max_widths[j])?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

impl std::ops::Add for Matrix {
    type Output = Result<Matrix, String>;

    fn add(self, other: Matrix) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrices have different dimensions".to_string());
        }

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }

        Ok(result)
    }
}

impl std::ops::Sub for Matrix {
    type Output = Result<Matrix, String>;

    fn sub(self, other: Matrix) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrices have different dimensions".to_string());
        }

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }

        Ok(result)
    }
}

impl std::ops::Mul<f64> for Matrix {
    type Output = Matrix;

    fn mul(self, scalar: f64) -> Self::Output {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] * scalar;
            }
        }
        result
    }
}

impl std::ops::Mul for Matrix {
    type Output = Result<Matrix, String>;

    fn mul(self, other: Matrix) -> Self::Output {
        if self.cols != other.rows {
            return Err("Matrices cannot be multiplied due to incompatible dimensions".to_string());
        }

        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    result.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }

        Ok(result)
    }
}

// fn main() {
//     let matrix1_str = "1 2 \n4 5 ";
//     let matrix2_str = "7 8\n   11 12";

//     let matrix1 = Matrix::from_str(matrix1_str).unwrap();
//     let matrix2 = Matrix::from_str(matrix2_str).unwrap();

//     println!("Matrix 1:\n{}", matrix1);
//     println!("Matrix 2:\n{}", matrix2);

//     match matrix1.clone() + matrix2.clone() {
//         Ok(result) => println!("Sum:\n{}", result),
//         Err(e) => println!("Error: {}", e),
//     }

//     match matrix1.clone() - matrix2.clone() {
//         Ok(result) => println!("Difference:\n{}", result),
//         Err(e) => println!("Error: {}", e),
//     }

//     match matrix1.clone() * matrix2.clone() {
//         Ok(result) => println!("Multiply:\n{}", result),
//         Err(e) => println!("Error: {}", e),
//     }

//     let matrix3_str = "1 2\n3 4";
//     let matrix3 = Matrix::from_str(matrix3_str).unwrap();

//     println!("Matrix 3:\n{}", matrix3);

//     match matrix3.determinant() {
//         Ok(det) => println!("Determinant of Matrix 3: {}", det),
//         Err(e) => println!("Error: {}", e),
//     }

//     println!("Transposed Matrix 3:\n{}", matrix3.transpose());

//     match matrix3.inverse() {
//         Ok(inverse) => {
//             println!("Inverse of Matrix 3:\n{}", inverse);
//             match matrix3.clone() * inverse {
//                 Ok(result) => println!("matrix one:\n{}", result),
//                 Err(e) => println!("Error: {}", e),
//             }
//         }
//         Err(e) => println!("Error: {}", e),
//     }

//     let scaled_matrix = matrix3.clone() * 2.5; // Умножение на константу
//     println!("Scaled Matrix 3 (by 2.5):\n{}", scaled_matrix);

//     let system_str = "2 3 5 10\n1 -1 2 3\n3 2 -1 4";
//     let system_matrix = Matrix::from_str(system_str).unwrap();

//     println!("System of equations:\n{}", system_matrix);

//     match system_matrix.gaussian_elimination() {
//         Ok(solutions) => {
//             println!("Solution using Gaussian elimination:");
//             for (i, &x) in solutions.iter().enumerate() {
//                 println!("x{} = {}", i + 1, x);
//             }
//         }
//         Err(e) => println!("Error: {}", e),
//     }

//     match system_matrix.cramer_rule() {
//         Ok(solutions) => {
//             println!("Solution using Cramer's rule:");
//             for (i, &x) in solutions.iter().enumerate() {
//                 println!("x{} = {}", i + 1, x);
//             }
//         }
//         Err(e) => println!("Error: {}", e),
//     }
// }