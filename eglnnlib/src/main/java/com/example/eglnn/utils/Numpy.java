package com.example.eglnn.utils;

public class Numpy {

    public static float[][] dot(float[][] mat1, float[][] mat2) {
        int mat1_row = mat1.length;
        int mat1_col = mat1[0].length;
        int mat2_row = mat2.length;
        int mat2_col = mat2[0].length;

        checkShape(mat1_col, mat2_row);
        float[][] result = new float[mat1_row][mat2_col];

        for (int row1 = 0; row1 < mat1_row; row1++) {
            for (int col1 = 0; col1 < mat1_col; col1++) {
                for (int col2 = 0; col2 < mat2_col; col2++) {
                    result[row1][col2] += mat1[row1][col1] * mat2[col1][col2];
                }
            }
        }
        return result;
    }

    private static void checkShape(int mat1_col, int mat2_row) {
        if (mat1_col != mat2_row) {
            try {
                throw new Exception("mat1 与 mat2 维度不同");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

}
