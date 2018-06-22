package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;
import com.example.cnnlib.utils.DataUtils;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.ComputeRender.initConvolutePro;

public class ConvolutionLayer2 extends Layer {

    private static final String TAG = "ConvolutionLayer";

    private Layer mPreLayer;
    private List<float[]> mKennels;
    private int[] mStrides;
    private int[] mKennelShape;
    private int mPadding;
    private int mShaderPro;
    private int mNumGroupsY;
    private int[] mParams;
    private NonLinearLayer.NonLinearType mType;

    private int mKennelTex;
    private int mKennelAttachId;


    public ConvolutionLayer2(Context context, Layer preLayer, int[] shape, int[] kennelShape, int padding, int[] strides, NonLinearLayer.NonLinearType type) {
        super(context, shape);
        this.mPreLayer = preLayer;
        this.mKennelShape = kennelShape;
        this.mPadding = padding;
        this.mStrides = strides;
        this.mType = type;
    }

    private void initConv() {
        int localSizeY = getCompShaderLocalSizeY(mOutputShape);
        mNumGroupsY = (int) Math.ceil(mOutputShape[1] * 1.0d / localSizeY);
        mShaderPro = initConvolutePro(mContext, "conv2.comp", mKennelShape, mOutputShape[0], localSizeY);
        mAttachID = AttachIDManager.getInstance().getDataAttachID();
        mOutTex = ComputeRender.createTexture();

        int[] kennelRestoreShape = calculateKennelRestoreShape();
        mKennels = createKennels();

        mKennelAttachId = AttachIDManager.getInstance().getConvAttachId();
        mKennelTex = ComputeRender.createConvKennelTexture();
        int[] kennelRestoreStartIndex = AttachIDManager.getInstance().getLastConvKennelIndex();
        transferKennelToTex(kennelRestoreShape, mKennelTex, kennelRestoreStartIndex[3]);
        int startY = kennelRestoreStartIndex[3];
        int endY = kennelRestoreStartIndex[3] + mOutputShape[0];
        AttachIDManager.getInstance().saveConvKennelIndex(new int[]{0, startY, 0, endY});

        mParams = new int[17];
        int[] inputShape = mPreLayer.getOutputShape();
        mParams[0] = mKennelShape[0];
        mParams[1] = mKennelShape[1];
        mParams[2] = mKennelShape[2];
        mParams[3] = inputShape[0];
        mParams[4] = inputShape[1];
        mParams[5] = inputShape[2];
        mParams[6] = mOutputShape[0];
        mParams[7] = mOutputShape[1];
        mParams[8] = mOutputShape[2];
        mParams[9] = mStrides[0];
        mParams[10] = mStrides[1];
        mParams[11] = mPadding;
        mParams[12] = mType.index;
        mParams[13] = kennelRestoreStartIndex[3];       // 起始点的y坐标
        mParams[14] = kennelRestoreShape[0];
        mParams[15] = kennelRestoreShape[1];
        mParams[16] = kennelRestoreShape[2];
    }

    private List<float[]> createKennels() {
        List<float[]> kennels = new ArrayList<>();
        for (int i = 0; i < mOutputShape[2]; i++) {
            kennels.add(DataUtils.createConvKennel(i + 1, 3, 3, 4, 1));
        }
        return kennels;
    }

    // kennel size 不能大于 4 * 8192
    private void transferKennelToTex(int[] kennelRestoreShape, int tex, int yOffset) {
        for (int i = 0; i < mKennels.size(); i++) {
            ComputeRender.transferToTexture(FloatBuffer.wrap(mKennels.get(i)), tex, 0, yOffset + i, kennelRestoreShape[0], 1);
        }
    }

    private int[] calculateKennelRestoreShape() {
        int[] kennelRestoreShape = new int[3];
        int[] inputShape = mPreLayer.getOutputShape();
        int inputSize = inputShape[0] * inputShape[1] * inputShape[2] + 1;  // 最后一列存储bias
        int width = inputSize / 4;
        int remain = inputSize % 4;
        kennelRestoreShape[0] = remain == 0 ? width : width + 1;
        kennelRestoreShape[1] = 1;
        kennelRestoreShape[2] = remain - 1 < 0 ? 3 : remain - 1;                    // 表示最后一位数据的下标
        return kennelRestoreShape;
    }

    @Override
    public void initialize() {
        initConv();
    }

    @Override
    protected void bindTextureAndBuffer() {
        ComputeRender.bindTextureAndBuffer(mKennelTex, mKennelAttachId);
        ComputeRender.bindTextureAndBuffer(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc() {
        ComputeRender.performConvolute(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mKennelTex, mNumGroupsY, mOutputShape[2]);
    }

}
