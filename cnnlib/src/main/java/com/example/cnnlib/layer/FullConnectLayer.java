package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;
import com.example.cnnlib.utils.DataUils;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.ComputeRender.initPoolingPro;

/**
 * kennel 存储方式为 纹理每行存储一个 kennel， 纹理高度与kennel个数相同
 * 全连接前接 flat 或 全连接层  只有纹理的第一个通道存储数据
 */
public class FullConnectLayer extends Layer {

    private static final String TAG = "FullConnectLayer";

    private Layer mPreLayer;
    private List<float[]> mKennels;
    private int mNumGroupsY;
    private int mShaderPro;
    private int mKennelAmount;                      // kennel组 个数

    private int[] mParams;
    private int mKennelTex;
    private int mKennelAttachID;

    public FullConnectLayer(Context context, Layer preLayer, int[] shape) {
        super(context, shape);
        this.mPreLayer = preLayer;
        this.mKennelAmount = shape[0] * shape[1];
        initFullConnect();
    }

    private void initFullConnect() {
        int localSizeY = getCompShaderLocalSizeY(mOutputShape);
        mNumGroupsY = (int) Math.ceil(mOutputShape[1] * 1.0d / localSizeY);
        mShaderPro = initPoolingPro(mContext, "full_connect.comp", mKennelAmount, mOutputShape[0], localSizeY);
        mAttachID = AttachIDManager.getInstance().getAttachID();
        mOutTex = ComputeRender.createTexture(mAttachID);


        int[] kennelSize = calculateKennelSize();
        mKennelAttachID = AttachIDManager.getInstance().getAttachID();
        mKennelTex = ComputeRender.createTexture(mKennelAttachID, kennelSize[0], mKennelAmount);
        mKennels = createKennels();
        transferKennelToTex(kennelSize, mKennelTex);

        int[] inputShape = mPreLayer.getOutputShape();

        mParams = new int[9];
        mParams[0] = inputShape[0];
        mParams[1] = inputShape[1];
        mParams[2] = inputShape[2];
        mParams[3] = mOutputShape[0];
        mParams[4] = mOutputShape[1];
        mParams[5] = mOutputShape[2];
        mParams[6] = kennelSize[0];
        mParams[7] = kennelSize[1];
        mParams[8] = kennelSize[2];
    }

    // kennel size width <= 1024 height=1
    private int[] calculateKennelSize() {
        int[] kennelSize = new int[3];
        int[] inputShape = mPreLayer.getOutputShape();
        int inputSize = inputShape[0] * inputShape[1] * inputShape[2] + 1;  // 最后一列存储bias
        int width = inputSize / 4;
        int remain = inputSize % 4;
        kennelSize[0] = remain == 0 ? width : width + 1;
        kennelSize[1] = 1;
        kennelSize[2] = remain - 1 < 0 ? 3 : remain - 1;                    // 表示最后一位数据的下标
        return kennelSize;
    }

    private void transferKennelToTex(int[] kennelSize, int tex) {
        for (int i = 0; i < mKennels.size(); i++) {
            ComputeRender.transferToTexture(FloatBuffer.wrap(mKennels.get(i)), tex, 0, i, kennelSize[0], 1);
        }
    }

    private List<float[]> createKennels() {
        List<float[]> kennels = new ArrayList<>();
        int[] inputShape = mPreLayer.getOutputShape();
        int inputSize = inputShape[0] * inputShape[1] * inputShape[2] + 1;     // 最后一列存储bias
        for (int i = 0; i < mKennelAmount; i++) {
            float[] kennel = DataUils.createFullConnKennel(inputSize, i);
            kennels.add(kennel);
        }
        return kennels;
    }

    @Override
    protected void bindTextureAndBuffer() {
        ComputeRender.bindTextureAndBuffer(mKennelTex, mKennelAttachID);
        ComputeRender.bindTextureAndBuffer(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc() {
        ComputeRender.performFullConnect(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mKennelTex, mNumGroupsY);
    }

    @Override
    public void readOutput() {

    }

}
