// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract LogStorage {
    struct Log {
        string action;
        string data;
        uint256 timestamp;
    }

    Log[] public logs;
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "فقط صاحب قرارداد می‌تواند این کار را انجام دهد");
        _;
    }

    function storeLog(string memory action, string memory data) public onlyOwner {
        logs.push(Log({
            action: action,
            data: data,
            timestamp: block.timestamp
        }));
    }

    function getLogCount() public view returns (uint256) {
        return logs.length;
    }

    function getLog(uint256 index) public view returns (string memory action, string memory data, uint256 timestamp) {
        require(index < logs.length, "لاگ با این شاخص وجود ندارد");
        Log memory log = logs[index];
        return (log.action, log.data, log.timestamp);
    }
}