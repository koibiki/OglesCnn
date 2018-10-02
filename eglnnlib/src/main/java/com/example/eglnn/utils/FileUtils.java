package com.example.eglnn.utils;

import android.content.Context;
import android.content.res.AssetManager;

import java.io.ByteArrayInputStream;
import java.io.Closeable;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class FileUtils {

    public static void copyToSD(Context context, String path, String fileName) {
        InputStream inputStream = null;
        FileOutputStream fos = null;
        try (AssetManager am = context.getAssets()) {
            inputStream = am.open(fileName);
            File file = new File(path, fileName);
            if (!file.getParentFile().exists()) {
                file.getParentFile().mkdirs();
            }
            if (!file.exists()) {
                fos = new FileOutputStream(file);
                int len;
                byte[] buffer = new byte[1024];
                while ((len = inputStream.read(buffer)) != -1) {
                    fos.write(buffer, 0, len);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            close(fos);
            close(inputStream);
        }
    }

    public static void saveToSD(byte[] bytes, String path, String fileName) {
        ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
        FileOutputStream fos = null;
        try {
            File file = new File(path, fileName);
            if (!file.getParentFile().exists()) {
                file.getParentFile().mkdirs();
            }
            if (!file.exists()) {
                fos = new FileOutputStream(file);
                int len;
                byte[] buffer = new byte[1024];
                while ((len = bais.read(buffer)) != -1) {
                    fos.write(buffer, 0, len);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            close(fos);
            close(bais);
        }
    }

    private static void close(Closeable closeable) {
        try {
            if (closeable != null) {
                closeable.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
