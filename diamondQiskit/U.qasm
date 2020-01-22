OPENQASM 2.0;
include "qelib1.inc";

gate dU(omega) qr {
    cx qr[1],qr[0];
    ch qr[0],qr[1];
    cx qr[1],qr[0];
    cz qr[2],qr[3];
    swap qr[2],qr[3];
    z qr[2];
    z qr[3];
    h qr[3];
    ccx qr[1],qr[2],qr[3];
    h qr[3];
    cswap qr[1],qr[2],qr[3];
    cz qr[1],qr[2];
    cz qr[1],qr[3];
    rz(omega) qr[1];
    h qr[3];
    ccx qr[0],qr[2],qr[3];
    h qr[3];
    cswap qr[0],qr[2],qr[3];
    z qr[0];
    rz(-omega) qr[0];
    cx qr[1],qr[0];
    ch qr[0],qr[1];
    cx qr[1],qr[0];
}

qreg qr[4];
dU(3.14) qr;