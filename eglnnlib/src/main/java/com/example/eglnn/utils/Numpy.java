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

    public static float[] argmax(float[] in) {
        float[] argmax = new float[2];
        argmax[0] = 0;
        argmax[1] = in[0];
        int len = in.length;
        for (int i = 1; i < len; i++) {
            if (in[i] > argmax[1]) {
                argmax[0] = i;
                argmax[1] = in[i];
            }
        }
        return argmax;
    }

    public static float[] argmin(float[] in) {
        float[] argmin = new float[2];
        argmin[0] = 0;
        argmin[1] = in[0];
        int len = in.length;
        for (int i = 1; i < len; i++) {
            if (in[i] < argmin[1]) {
                argmin[0] = i;
                argmin[1] = in[i];
            }
        }
        return argmin;
    }

    public static float[][] transpose(float[][] mat) {
        int mat_col = mat[0].length;
        int mat_row = mat.length;
        float[][] new_mat = new float[mat_col][mat_row];

        for (int col = 0; col < mat_col; col++) {
            for (int row = 0; row < mat_row; row++) {
                new_mat[col][row] = mat[row][col];
            }
        }
        return new_mat;
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
