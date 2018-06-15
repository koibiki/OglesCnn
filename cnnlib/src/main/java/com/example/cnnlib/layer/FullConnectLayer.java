package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.model.LayerParams;
import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;
import com.example.cnnlib.utils.DataUils;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.ComputeRender.initPoolingPro;

public class FullConnectLayer extends Layer {

    private static final String TAG = "FullConnectLayer";

    List<float[]> mKennels = new ArrayList<>();
    private int mNumGroupsY;
    private int mShaderPro;
    private int mKennelAmount;                      // kennel组 个数
    private int[] mKennelSize = new int[3];         // 每个kennel的大小 w,h,c
    private int[] mParams;
    private int mKennelTex;

    public FullConnectLayer(Context context, LayerParams layerParams, int kennelAmount) {
        super(context, layerParams);
        this.mKennelAmount = kennelAmount;

        initFullConnect();
    }

    private void initFullConnect() {
        int localSizeY = getCompShaderLocalSizeY(mLayerParams.outputShape);
        mNumGroupsY = (int) Math.ceil(mLayerParams.outputShape[1] * 1.0d / localSizeY);
        mShaderPro = initPoolingPro(mContext, "full_connect.comp", mKennelAmount, mLayerParams.outputShape[0], localSizeY);
        mAttachID = AttachIDManager.getInstance().getAttachID();
        mOutTex = ComputeRender.createTexture(mAttachID);
        int attachID = AttachIDManager.getInstance().getAttachID();
        mKennelTex = ComputeRender.createTexture(attachID);

        calculateKennelSize();
        createKennels();
        transferKennelToTex(mKennelTex);

        mParams = new int[9];
        mParams[0] = mLayerParams.inputShape[0];
        mParams[1] = mLayerParams.inputShape[1];
        mParams[2] = mLayerParams.inputShape[2];
        mParams[3] = mLayerParams.outputShape[0];
        mParams[4] = mLayerParams.outputShape[1];
        mParams[5] = mLayerParams.outputShape[2];
        mParams[6] = mKennelSize[0];
        mParams[7] = mKennelSize[1];
        mParams[8] = mKennelSize[2];
    }

    // kennel size width <= 1024 height=1
    private void calculateKennelSize() {
        int[] inputShape = mLayerParams.inputShape;
        int inputSize = inputShape[0] * inputShape[1] * inputShape[2] + 1;  // 最后一列存储bias
        int width = inputSize / 4;
        int remain = inputSize % 4;
        mKennelSize[0] = remain == 0 ? width : width + 1;
        mKennelSize[1] = 1;
        mKennelSize[2] = remain - 1 <0? 3: remain-1;    // 表示最后一位数据的下标
    }

    private void transferKennelToTex(int tex) {
        int cols = 0;
        for (int i = 0; i < mKennels.size(); i++) {
            int startX = cols / 1024 * mKennelSize[0];
            int startY = i % 1024;
            ComputeRender.transferToTexture(FloatBuffer.wrap(mKennels.get(i)), tex, startX, startY, mKennelSize[0], mKennelSize[1]);
        }
    }

    private void createKennels() {
        int[] inputShape = mLayerParams.inputShape;
        int inputSize = inputShape[0] * inputShape[1] * inputShape[2] + 1;     // 最后一列存储bias
        for (int i = 0; i < mKennelAmount; i++) {
            float[] kennel = DataUils.createFullConnKennel(inputSize, i);
            mKennels.add(kennel);
        }
    }

    @Override
    public int forwardProc(int inTex) {
        ComputeRender.performFullConnect(mShaderPro, mParams, inTex, mOutTex, mKennelTex, mNumGroupsY);
        return mOutTex;
    }

    @Override
    public void readOutput() {

    }

}
