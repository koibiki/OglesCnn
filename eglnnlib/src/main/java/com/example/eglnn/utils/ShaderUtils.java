package com.example.eglnn.utils;

import android.content.res.Resources;
import android.opengl.GLES31;

import java.io.InputStream;

public class ShaderUtils {

    private static final String TAG = "ShaderUtils";

    public static void vglAttachShaderSource(int prog, int type, String source) {
        int sh = GLES31.glCreateShader(type);
        GLES31.glShaderSource(sh, source);
        GLES31.glCompileShader(sh);
        GLES31.glAttachShader(prog, sh);
        GLES31.glDeleteShader(sh);
    }

    public static String loadFromAssetsFile(String fname, Resources res) {
        StringBuilder result = new StringBuilder();
        try {
            InputStream is = res.getAssets().open(fname);
            int ch;
            byte[] buffer = new byte[1024];
            while (-1 != (ch = is.read(buffer))) {
                result.append(new String(buffer, 0, ch));
            }
        } catch (Exception e) {
            return null;
        }
        return result.toString().replaceAll("\\r\\n", "\n");
    }

}
