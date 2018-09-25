package com.example.eglnn;

import android.content.Context;
import android.opengl.GLES31;
import android.os.Environment;
import android.util.Log;

import com.example.eglnn.eglenv.GLES31BackEnv;
import com.example.eglnn.layer.Layer;
import com.example.eglnn.utils.DataUtils;

import java.util.ArrayList;

import static com.example.eglnn.utils.Constants.S_TEXTURE_SIZE;

public class NnNetwork {

    private static final String TAG = "CnnNetwork";

    private Context mContext;
    private ArrayList<Layer> mLayers;

    private volatile boolean mIsInit = false;
    private GLES31BackEnv mGLes31BackEnv;
    private String mNetStructureFile;
    private String mRootDir = Environment.getExternalStorageDirectory().getAbsolutePath();


    public NnNetwork(Context context) {
        this.mContext = context;
        this.mLayers = new ArrayList<>();
        this.mGLes31BackEnv = new GLES31BackEnv(S_TEXTURE_SIZE, S_TEXTURE_SIZE);
    }

    public void addLayer(Layer layer) {
        mLayers.add(layer);
    }

    public void initialize() {
        mGLes31BackEnv.post(this::actualInitialize);
    }

    public void predict(float[][] input) {
        mGLes31BackEnv.post(() -> actualPredict(input));
    }

    private void actualPredict(float[][] input) {
        Log.w(TAG, "actualPredict");
        runNet(input);
        actualReadOutput();
    }

    private void actualInitialize() {
        if (!mIsInit) {
            mIsInit = true;
            for (Layer layer : mLayers) {
                layer.initialize();
            }
        }
    }

    private void runNet(float[][] input) {
        long begin = System.currentTimeMillis();
        int count = 100;
        int filter = 10;
        for (int ii = 0; ii < count + filter; ii++) {
            long begin1 = System.currentTimeMillis();
            if (ii == filter) {
                begin = begin1;
            }
            int size = mLayers.size();
            for (int i = 0; i < size; i++) {
                mLayers.get(i).forwardProc(input, true);
            }
            actualReadOutput();
            Log.w(TAG, "spent:" + (System.currentTimeMillis() - begin1));
        }
        Log.w(TAG, "ave spent:" + (System.currentTimeMillis() - begin) * 1.0f / count);
    }

    public void readOutput() {
        mGLes31BackEnv.post(this::actualReadOutput);
    }

    private void actualReadOutput() {
        Layer layer = mLayers.get(mLayers.size() - 1);
        DataUtils.readOutput(layer);
    }
}
