package com.example.eglnn.utils;

import android.content.Context;

import org.msgpack.MessagePack;
import org.msgpack.packer.BufferPacker;
import org.msgpack.unpacker.BufferUnpacker;

import java.io.ByteArrayInputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;

public class MessagePackUtils {

    public static Object unpackParam(Context context, String fileName, Class classType) throws IOException {
        MessagePack msgpack = new MessagePack();
        byte[] bytes = getParamsBytes(context, fileName);
        if (bytes != null) {
            BufferUnpacker bufferUnpacker = msgpack.createBufferUnpacker(bytes);
            return bufferUnpacker.read(classType);
        } else {
            return null;
        }
    }

    public static Object[] unpackParam(Context context, String fileName, Class[] classTypes) {
        MessagePack msgpack = new MessagePack();
        Object[] objects = null;
        try {
            objects = new Object[2];
            byte[] bytes = getParamsBytes(context, fileName);
            BufferUnpacker bufferUnpacker = msgpack.createBufferUnpacker(bytes);
            objects[0] = bufferUnpacker.read(classTypes[0]);
            objects[1] = bufferUnpacker.read(classTypes[1]);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return objects;
    }

    public static void test(Context context) {
        MessagePack messagePack = new MessagePack();
        try {
            BufferPacker bufferPacker = messagePack.createBufferPacker();
            bufferPacker.write(new int[]{1, 2, 3});
            bufferPacker.write(new float[]{1.0f, 2.0f, 3.1f});
            byte[] bytes = bufferPacker.toByteArray();

            BufferUnpacker bufferUnpacker = messagePack.createBufferUnpacker(bytes);
            int[] read = bufferUnpacker.read(int[].class);
            float[] read2 = bufferUnpacker.read(float[].class);
            int a = read[0];
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static byte[] getParamsBytes(Context context, String fileName) {
        InputStream ins = null;
        ByteArrayInputStream in = null;
        byte[] bytes = null;
        try {
            ins = context.getAssets().open(fileName);
            int available = ins.available();
            bytes = new byte[available];
            ins.read(bytes);
            in = new ByteArrayInputStream(bytes);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            close(ins);
            close(in);
        }
        return bytes;
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
