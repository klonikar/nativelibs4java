package com.nativelibs4java.mono.bridj;
import org.bridj.Pointer;
import org.bridj.StructObject;
import org.bridj.ann.Field;
import org.bridj.ann.Library;
/**
 * <i>native declaration : mono/metadata/opcodes.h</i><br>
 * This file was autogenerated by <a href="http://jnaerator.googlecode.com/">JNAerator</a>,<br>
 * a tool written by <a href="http://ochafik.free.fr/">Olivier Chafik</a> that <a href="http://code.google.com/p/jnaerator/wiki/CreditsAndLicense">uses a few opensource projects.</a>.<br>
 * For help, please visit <a href="http://nativelibs4java.googlecode.com/">NativeLibs4Java</a> or <a href="http://bridj.googlecode.com/">BridJ</a> .
 */
@Library("mono") 
public class MonoOpcode extends StructObject {
	public MonoOpcode() {
		super();
	}
	public MonoOpcode(Pointer pointer) {
		super(pointer);
	}
	@Field(0) 
	public byte argument() {
		return this.io.getByteField(this, 0);
	}
	@Field(0) 
	public MonoOpcode argument(byte argument) {
		this.io.setByteField(this, 0, argument);
		return this;
	}
	@Field(1) 
	public byte flow_type() {
		return this.io.getByteField(this, 1);
	}
	@Field(1) 
	public MonoOpcode flow_type(byte flow_type) {
		this.io.setByteField(this, 1, flow_type);
		return this;
	}
	@Field(2) 
	public short opval() {
		return this.io.getShortField(this, 2);
	}
	@Field(2) 
	public MonoOpcode opval(short opval) {
		this.io.setShortField(this, 2, opval);
		return this;
	}
}