package com.example.eglnn.utils;

import android.content.Context;

import org.msgpack.MessagePack;
import org.msgpack.unpacker.BufferUnpacker;

import java.io.ByteArrayInputStream;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;

public class MessagePackUtils {


    public static Object unpackParam(Context context, String fileName, Class classType) throws IOException {
        MessagePack msgpack = new MessagePack();
        byte[] bytes = getParamsBytes(context, fileName);
        BufferUnpacker bufferUnpacker = msgpack.createBufferUnpacker(bytes);
        return bufferUnpacker.read(classType);
    }

    private static byte[] getParamsBytes(Context context, String fileName) throws IOException {
        InputStream ins = context.getAssets().open(fileName);
        int available = ins.available();
        byte[] bytes = new byte[available];
        ins.read(bytes);
        close(ins);
        ByteArrayInputStream in = new ByteArrayInputStream(bytes);
        close(in);
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
